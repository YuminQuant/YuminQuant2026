"""Offline regression distillation from boosted trees to a tabular MLP."""
from contextlib import ExitStack, contextmanager
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from yq_ml_alpha.models.torch_support import (
    TorchAlphaModel, _cpu_state_dict, _device, _diagnostics, _set_seed, _torch_modules,
)
from yq_ml_alpha.models.xgb_training import make_matrix
from yq_ml_alpha.output import artifact_io


def _params(raw):
    params = dict(raw)
    for key in ("epochs", "patience", "batch_size"):
        if type(params[key]) is not int or params[key] < 1:
            raise ValueError(f"KD-Ensemble {key} must be a positive integer")
    widths = params["hidden_layers"]
    if not isinstance(widths, list) or not widths or any(type(width) is not int or width < 1 for width in widths):
        raise ValueError("KD-Ensemble requires positive student hidden widths")
    if not 0 < params["distill_weight"] < 1 or not 0 <= params["dropout"] < 1:
        raise ValueError("invalid KD-Ensemble distillation weight or student dropout")
    if any(not np.isfinite(params[key]) or params[key] <= 0 for key in ("lr", "smooth_l1_beta")):
        raise ValueError("KD-Ensemble requires positive learning rate and Smooth L1 beta")
    if params["teacher"].get("objective") != "reg:squarederror":
        raise ValueError("KD-Ensemble teacher must use squared-error regression")
    return params


def _student(nn, width, params):
    layers = []
    for hidden in params["hidden_layers"]:
        layers.extend((nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(params["dropout"])))
        width = hidden
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


def _features(frame, columns):
    return np.ascontiguousarray(frame[columns].to_numpy(dtype=np.float32, copy=False))


def _labeled(frame, context):
    targets = frame[context.label_column].to_numpy(dtype=np.float32, na_value=np.nan)
    valid = np.isfinite(targets)
    if not valid.any():
        raise ValueError("KD-Ensemble requires nonempty train and validation sets")
    return _features(frame.loc[valid], context.feature_columns), targets[valid]


def _teacher_scores(features, labels, params):
    import xgboost as xgb

    native = dict(params)
    rounds = native.pop("n_estimators")
    threads = native.pop("n_jobs", -1)
    native["nthread"] = threads
    native["seed"] = native.pop("random_state", 0)
    matrix = make_matrix(xgb, features, labels, n_jobs=threads)
    teacher = xgb.train(native, matrix, num_boost_round=rounds, verbose_eval=False)
    return teacher.predict(matrix).astype("float32", copy=False)


class KDEnsembleAlphaModel(TorchAlphaModel):
    prediction_replay_safe = True
    prediction_inputs_readonly = True

    @staticmethod
    def adjust_training_window(config, calendar, window):
        from yq_ml_alpha.features.kd_components import adjust_student_window
        return adjust_student_window(config, calendar, window)

    @staticmethod
    def training_stage_path(config, window_id, stage):
        from yq_ml_alpha.features.kd_components import branch_path
        if stage == "distill":
            return config.model.artifact_dir / window_id
        return branch_path(config, window_id, stage)

    @staticmethod
    def execute_training_stage(config, window_id, stage, run_id, *, resume=False):
        from yq_ml_alpha.features.kd_components import train_stage
        return train_stage(config, window_id, stage, run_id, resume=resume)

    def capture_prediction_weights(self, provider, dates):
        self.component_weights = provider.checkpoint()

    @contextmanager
    def prediction_session(self):
        with super().prediction_session(), ExitStack() as sessions:
            self._component_sessions = sessions
            self._component_models = None
            try:
                yield self
            finally:
                self._component_models = self._component_sessions = None

    def prediction_input_scope(self, provider):
        if self._component_models is None:
            self._component_models = provider.restore_components(self.component_weights, self._component_sessions)
        return provider.prediction_components(self._component_models)

    def fit(self, train_data, valid_data, context):
        torch, nn = _torch_modules()
        self.params = _params(context.model_params)
        self.feature_columns = list(context.feature_columns)
        self.input_dim = len(self.feature_columns)
        if self.input_dim != 192 or context.model_search:
            raise ValueError("KD-Ensemble requires 192 weak factors and fixed model parameters")
        if train_data.empty or valid_data.empty or train_data.trade_date.max() >= valid_data.trade_date.min():
            raise ValueError("KD-Ensemble requires separate, chronological train and validation sets")
        train_x, train_y = _labeled(train_data, context)
        valid_x, valid_y = _labeled(valid_data, context)
        diagnostics = _diagnostics(context)
        started = perf_counter()
        soft = _teacher_scores(train_x, train_y, self.params["teacher"])
        if not np.isfinite(soft).all():
            raise ValueError("KD-Ensemble teacher produced nonfinite scores")
        if diagnostics["print_epoch"]:
            print(f"kd-ensemble teacher_rows={len(train_x)} elapsed_s={perf_counter() - started:.2f}", flush=True)
        train_x, train_y, soft, valid_x, valid_y = map(torch.from_numpy, (train_x, train_y, soft, valid_x, valid_y))
        _set_seed(torch, self.params["seed"])
        device = _device(torch, self.params["device"])
        self.model = _student(nn, self.input_dim, self.params).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params["lr"])
        mse = nn.MSELoss()
        smooth = nn.SmoothL1Loss(beta=self.params["smooth_l1_beta"])
        weight = self.params["distill_weight"]
        batch_size = self.params["batch_size"]
        best, best_state, best_epoch, stale = np.inf, None, 0, 0
        self.loss_history = []
        for epoch in range(1, self.params["epochs"] + 1):
            self.model.train()
            totals = torch.zeros(3, device=device)
            for indices in torch.randperm(len(train_x)).split(batch_size):
                optimizer.zero_grad(set_to_none=True)
                prediction = self.model(train_x[indices].to(device)).squeeze(1)
                hard_loss = mse(prediction, train_y[indices].to(device))
                distill_loss = smooth(prediction, soft[indices].to(device))
                loss = (1 - weight) * hard_loss + weight * distill_loss
                loss.backward()
                optimizer.step()
                totals += torch.stack((loss.detach(), hard_loss.detach(), distill_loss.detach())) * len(indices)
            train_loss, train_mse, train_smooth = (totals / len(train_x)).tolist()
            self.model.eval()
            with torch.no_grad():
                error = torch.zeros((), device=device)
                for x, y in zip(valid_x.split(batch_size), valid_y.split(batch_size)):
                    error += (self.model(x.to(device)).squeeze(1) - y.to(device)).square().sum()
                valid_mse = float(error / len(valid_x))
            if not np.isfinite([train_loss, valid_mse]).all():
                raise ValueError(f"nonfinite KD-Ensemble loss at epoch {epoch}")
            improved = valid_mse < best
            if improved:
                best, best_state, best_epoch, stale = valid_mse, _cpu_state_dict(self.model), epoch, 0
            else:
                stale += 1
            self.loss_history.append(dict(window_id=context.artifact_dir.name, epoch=epoch,
                train_loss=train_loss, train_mse=train_mse, train_smooth_loss=train_smooth,
                valid_loss=valid_mse, best_loss=best, is_best=improved, stale_epochs=stale,
                elapsed_seconds=perf_counter() - started))
            if diagnostics["print_epoch"]:
                print(f"kd-ensemble epoch={epoch} train_loss={train_loss:.6g} train_mse={train_mse:.6g} "
                      f"train_smooth_loss={train_smooth:.6g} valid_mse={valid_mse:.6g} "
                      f"stale={stale}/{self.params['patience']}", flush=True)
            if stale >= self.params["patience"]:
                break
        self.model.load_state_dict(best_state)
        self.model.to("cpu").eval()
        self.model_info = dict(params=self.params, features=self.feature_columns, selection="validation_mse",
                               train_rows=len(train_x), valid_rows=len(valid_x), best_epoch=best_epoch,
                               best_valid_mse=best, teacher_target="training_row_prediction")

    def predict(self, data, context):
        if self.model is None or list(context.feature_columns) != self.feature_columns:
            raise ValueError("KD-Ensemble requires a fitted checkpoint with the same feature columns")
        torch, _ = _torch_modules()
        device = _device(torch, self.params["device"])
        self.model.to(device).eval()
        output = np.empty(len(data), dtype=np.float32)
        try:
            with torch.no_grad():
                values = torch.from_numpy(_features(data, self.feature_columns))
                for start in range(0, len(values), self.params["batch_size"]):
                    batch = values[start:start + self.params["batch_size"]]
                    output[start:start + len(batch)] = self.model(batch.to(device)).squeeze(1).cpu().numpy()
            if not np.isfinite(output).all():
                raise ValueError("KD-Ensemble student produced nonfinite scores")
            return pd.Series(output, index=data.index)
        finally:
            if not getattr(self, "_prediction_active", False):
                self.model.to("cpu")

    def save(self, path):
        path = Path(path)
        artifact_io.mkdir(path.parent, parents=True, exist_ok=True)
        artifact_io.torch_save(dict(kind="kd_ensemble", version=3, params=self.params,
            feature_columns=self.feature_columns, state_dict=_cpu_state_dict(self.model),
            components=self.component_weights, model_info=self.model_info, loss_history=self.loss_history), path)

    @classmethod
    def load(cls, path):
        _, nn = _torch_modules()
        saved = artifact_io.torch_load(Path(path), map_location="cpu")
        if saved.get("kind") != "kd_ensemble" or saved.get("version") != 3:
            raise ValueError("unsupported KD-Ensemble checkpoint")
        instance = cls()
        instance.params = _params(saved["params"])
        instance.feature_columns = list(saved["feature_columns"])
        instance.input_dim = len(instance.feature_columns)
        if instance.input_dim != 192:
            raise ValueError("invalid KD-Ensemble checkpoint feature width")
        instance.model = _student(nn, instance.input_dim, instance.params)
        instance.model.load_state_dict(saved["state_dict"])
        instance.model.eval()
        instance.component_weights = saved["components"]
        instance.model_info, instance.loss_history = saved["model_info"], saved["loss_history"]
        return instance
