from __future__ import annotations

import time

import numpy as np
import pandas as pd

from yq_ml_alpha.models.bar_gru_model import (
    BarGRUAlphaModel, _bar_tensor_batch, _cpu_state_dict, _date_groups,
    _device, _diagnostics, _negative_ic_loss, _release_torch_memory,
    _require_bar_bundle_tensor, _set_seed, _torch_modules,
)


class CrossSectionBarAlphaModel(BarGRUAlphaModel):
    """Encode bars in chunks, then train/predict one complete date graph at a time.

    Regressors implement encode(x) and forward_encoded(h). No graph is built
    over arbitrary stock minibatches or across different sample dates.
    """

    def fit(self, train_data, valid_data, context):
        raise ValueError("cross-section models require full-cross-section DatasetBundle inputs")

    def predict(self, data, context):
        raise ValueError("cross-section models require full-cross-section DatasetBundle inputs")

    @staticmethod
    def _check_bundle(bundle):
        if bundle is None or not bundle.full_cross_section:
            raise ValueError("cross-section model requires features.params.full_cross_section=true")
        if bundle.frame.duplicated(["trade_date", "ts_code"]).any():
            raise ValueError("duplicate stock/date nodes in cross-section bundle")
        if not bundle.frame.index.equals(pd.RangeIndex(len(bundle.frame))):
            raise ValueError("cross-section bundle rows must align with its tensor positions")

    def fit_bundle(self, train_bundle, valid_bundle, context):
        self._check_bundle(train_bundle)
        self._check_bundle(valid_bundle)
        super().fit_bundle(train_bundle, valid_bundle, context)

    def _encode_rows(self, torch, frame, tensor, rows, context, device):
        from torch.utils.checkpoint import checkpoint

        encoded = []
        # A scalar grad input enables reentrant checkpointing without keeping
        # all raw bar tensors on GPU. RNG state is preserved for GRU dropout.
        token = torch.zeros((), device=device, requires_grad=True)

        def encode(host, anchor):
            return self.model.encode(host.to(device)) + anchor * 0.0

        for start in range(0, len(rows), self.params["batch_size"]):
            selected = rows[start:start + self.params["batch_size"]]
            host = torch.from_numpy(_bar_tensor_batch(frame, context.feature_columns, selected, self.params, tensor))
            if self.model.training and self.params["gradient_checkpointing"]:
                encoded.append(checkpoint(encode, host, token, preserve_rng_state=True))
            else:
                encoded.append(self.model.encode(host.to(device)))
        return torch.cat(encoded, dim=0)

    def _training_loss_encoded(self, torch, encoded, target):
        return _negative_ic_loss(torch, self.model.forward_encoded(encoded), target)

    def _eval_full_dates(self, torch, frame, tensor, groups, context, device):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for rows in groups.values():
                if len(rows) < 2:
                    continue
                target = torch.from_numpy(frame.loc[rows, context.label_column].to_numpy(dtype="float32")).to(device)
                if int(torch.isfinite(target).sum()) < 2:
                    continue
                encoded = self._encode_rows(torch, frame, tensor, rows, context, device)
                score = self.model.forward_encoded(encoded)
                if not bool(torch.isfinite(score).all()):
                    raise ValueError("non-finite graph validation scores")
                loss = _negative_ic_loss(torch, score, target)
                if loss is not None:
                    losses.append(float(loss.item()))
        return float(np.mean(losses)) if losses else None

    def _fit(self, train_data, valid_data, context, *, train_tensor, valid_tensor):
        torch, nn = _torch_modules()
        self.params = self._normalize_params(context.model_params)
        if train_data.empty or valid_data.empty:
            raise ValueError("cross-section training requires non-empty training and validation data")
        if train_data.trade_date.max() >= valid_data.trade_date.min():
            raise ValueError("training dates must precede validation dates")
        _set_seed(torch, self.params["seed"])
        device = _device(torch, self.params["device"])
        self.model = self._build_regressor(nn).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params["lr"],
                                     weight_decay=self.params["weight_decay"])
        train_groups, valid_groups = _date_groups(train_data), _date_groups(valid_data)
        diagnostics = _diagnostics(context)
        best_state, best_loss, best_epoch, stale = None, float("inf"), 0, 0
        self.loss_history = []
        started = time.perf_counter()
        for epoch in range(1, self.params["epochs"] + 1):
            self.model.train()
            dates = list(train_groups)
            np.random.default_rng(self.params["seed"] + epoch).shuffle(dates)
            losses = []
            for date in dates:
                rows = train_groups[date]
                target = torch.from_numpy(train_data.loc[rows, context.label_column].to_numpy(dtype="float32")).to(device)
                if len(rows) < 2 or int(torch.isfinite(target).sum()) < 2:
                    continue
                optimizer.zero_grad(set_to_none=True)
                encoded = self._encode_rows(torch, train_data, train_tensor, rows, context, device)
                loss = self._training_loss_encoded(torch, encoded, target)
                if loss is None:
                    del encoded, target
                    continue
                if not bool(torch.isfinite(loss)):
                    raise ValueError(f"non-finite graph loss on {date}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params["grad_clip"], error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach().item()))
                del encoded, loss, target
            if not losses:
                raise ValueError("no usable training cross sections")
            valid_loss = self._eval_full_dates(torch, valid_data, valid_tensor, valid_groups, context, device)
            if valid_loss is None:
                raise ValueError("no usable validation cross sections")
            is_best = valid_loss + 1e-12 < best_loss
            if is_best:
                best_loss, best_epoch, stale = valid_loss, epoch, 0
                best_state = _cpu_state_dict(self.model)
            else:
                stale += 1
            self.loss_history.append({
                "window_id": context.artifact_dir.name, "epoch": epoch,
                "train_loss": float(np.mean(losses)), "valid_loss": valid_loss,
                "best_loss": best_loss, "is_best": is_best, "stale_epochs": stale,
                "elapsed_seconds": time.perf_counter() - started, "device": str(device),
                "model_class": self.__class__.__name__,
            })
            if diagnostics["enabled"] and diagnostics["print_epoch"]:
                print(f"window={context.artifact_dir.name} epoch={epoch} "
                      f"graph_train_loss={np.mean(losses):.6g} valid_loss={valid_loss:.6g} "
                      f"best={best_loss:.6g} patience={stale}/{self.params['patience']}", flush=True)
            if self.params["patience"] > 0 and stale >= self.params["patience"]:
                break
        self.model.load_state_dict(best_state)
        self.model.to("cpu")
        del optimizer
        _release_torch_memory(torch)
        self.model_info = {
            "window_id": context.artifact_dir.name, "alpha_id": context.alpha_id,
            "model_class": self.__class__.__name__, "device": str(device),
            "train_rows": len(train_data), "valid_rows": len(valid_data),
            "train_labeled_rows": int(np.isfinite(train_data[context.label_column]).sum()),
            "valid_labeled_rows": int(np.isfinite(valid_data[context.label_column]).sum()),
            "train_dates": len(train_groups), "valid_dates": len(valid_groups),
            "epochs_run": len(self.loss_history), "best_epoch": best_epoch, "best_loss": best_loss,
            "graph_scope": "all_eligible_stocks_per_date_including_unlabeled_nodes",
            "validation_loss": "negative_full_date_pearson_ic",
            "encoder_chunk_size": self.params["batch_size"],
            "graph_chunk_size": self.params["graph_chunk_size"],
            "gradient_checkpointing": self.params["gradient_checkpointing"],
        }

    def predict_bundle(self, bundle, context):
        self._check_bundle(bundle)
        return self._predict_tensor(bundle.frame, _require_bar_bundle_tensor(bundle), context)

    def _predict_tensor(self, data, tensor, context):
        if self.model is None:
            raise RuntimeError("model is not fitted")
        torch, _ = _torch_modules()
        device = _device(torch, self.params["device"])
        self.model.to(device)
        self.model.eval()
        scores = pd.Series(index=data.index, dtype="float32")
        try:
            with torch.no_grad():
                for rows in _date_groups(data).values():
                    encoded = self._encode_rows(torch, data, tensor, rows, context, device)
                    pred = self.model.forward_encoded(encoded)
                    if not bool(torch.isfinite(pred).all()):
                        raise ValueError("non-finite graph prediction scores")
                    scores.loc[rows] = pred.cpu().numpy().astype("float32")
        finally:
            if not getattr(self, "_prediction_active", False):
                self.model.to("cpu")
                _release_torch_memory(torch)
        return scores
