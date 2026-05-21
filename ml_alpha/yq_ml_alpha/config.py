from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple

try:
    import tomllib as _tomllib

    _TOML_BINARY_LOAD = True
except ModuleNotFoundError:  # pragma: no cover
    try:
        import tomli as _tomllib

        _TOML_BINARY_LOAD = True
    except ModuleNotFoundError:
        import toml as _tomllib

        _TOML_BINARY_LOAD = False


DateRange = Tuple[int, int]
OptionalDateRange = Optional[DateRange]
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DatesConfig:
    train: DateRange
    valid: OptionalDateRange = None
    predict: OptionalDateRange = None


@dataclass(frozen=True)
class SampleConfig:
    train_frequency: str
    predict_frequency: str | None = None


@dataclass(frozen=True)
class TrainSchemeConfig:
    type: str = "static"
    refit_frequency: str = "monthly"
    min_train_days: int = 756
    valid_days: int = 252
    train_lookback: str | None = None
    train_sample_count: int = 0
    validation_sample_count: int = 0
    validation_ratio: float | None = None


@dataclass(frozen=True)
class PreprocessConfig:
    cross_section_transform: str = "none"
    feature_fill_value: float = 0.0


@dataclass(frozen=True)
class PostprocessConfig:
    pass


@dataclass(frozen=True)
class LabelConfig:
    id: str
    root: Path = Path("data/label/stock/daily")


@dataclass(frozen=True)
class UniverseConfig:
    id: str = "mkt_all"


@dataclass(frozen=True)
class FiltersConfig:
    exclude_limit: bool = True
    exclude_st: bool = True
    exclude_bj: bool = True


@dataclass(frozen=True)
class FeaturesConfig:
    type: str
    root: Path
    columns: list[str] | str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MaterializeConfig:
    cache_samples: bool = False
    cache_dir: Path = Path("data/model_workspace/cache")
    predict_batch_size: int = 20


@dataclass(frozen=True)
class DiagnosticsConfig:
    enabled: bool = False
    print_epoch: bool = False
    write_loss_history: bool = False
    write_model_info: bool = False
    write_window_summary: bool = False


@dataclass(frozen=True)
class ModelConfig:
    name: str
    class_path: str
    artifact_dir: Path
    params: dict[str, Any] = field(default_factory=dict)
    search: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OutputConfig:
    kind: str = "signal"
    id: str = ""
    root: Path = Path("data/models")
    asset: str = "stock"
    frequency: str = "daily"
    base_root: Path = Path("data/stock_data/daily/pv")
    write_workers: int = 4
    write_metadata: bool = True


@dataclass(frozen=True)
class MlAlphaConfig:
    run_id: str
    alpha_id: str
    dates: DatesConfig
    sample: SampleConfig
    train_scheme: TrainSchemeConfig
    label: LabelConfig
    universe: UniverseConfig
    filters: FiltersConfig
    preprocess: PreprocessConfig
    postprocess: PostprocessConfig
    features: FeaturesConfig
    materialize: MaterializeConfig
    diagnostics: DiagnosticsConfig
    model: ModelConfig
    output: OutputConfig
    factor_id: str | None = None
    data_root: Path = Path("data")
    output_root: Path = Path("data/models")


def load_config(path: str | Path) -> MlAlphaConfig:
    config_path = Path(path)
    if _TOML_BINARY_LOAD:
        with config_path.open("rb") as file:
            raw = _tomllib.load(file)
    else:
        raw = _tomllib.load(str(config_path))

    data_root = _project_path(raw.get("data_root", "data"))
    factor_id = raw.get("factor_id")
    output = raw.get("output", {})
    output_kind = str(output.get("kind", "factor" if factor_id else "signal")).strip().lower()
    output_id = str(output.get("id", factor_id or raw.get("alpha_id", ""))).strip()
    run_id = str(raw.get("run_id", factor_id or "")).strip()
    alpha_id = str(raw.get("alpha_id", output_id)).strip()
    if not run_id:
        run_id = _required(raw, "run_id")
    if not alpha_id:
        alpha_id = _required(raw, "alpha_id")
    if not output_id:
        output_id = alpha_id
    if factor_id is not None:
        factor_id = str(factor_id).strip()
        if not factor_id:
            raise ValueError("factor_id cannot be empty")
        if output_id != factor_id:
            raise ValueError("factor output.id must match factor_id")
        if run_id.startswith("mdl_") or alpha_id.startswith("mdl_"):
            raise ValueError("factor configs must not use mdl_* run_id or alpha_id")
        if factor_id.startswith("e2e_fct_"):
            raise ValueError("factor configs must use semantic factor_id, not e2e_fct_*")
    default_output_root = "data/factors" if output_kind == "factor" else "data/models"
    output_root = _project_path(output.get("root", raw.get("output_root", default_output_root)))

    dates = raw.get("dates", {})
    label = raw.get("label", {})
    universe = raw.get("universe", {})
    filters = raw.get("filters", {})
    preprocess = raw.get("preprocess", {})
    postprocess = raw.get("postprocess", {})
    features = raw.get("features", {})
    materialize = raw.get("materialize", {})
    diagnostics = raw.get("diagnostics", {})
    model = raw.get("model", {})

    model_params = dict(model.get("params", {}))
    legacy_search = model_params.pop("search", {})
    model_search = dict(model.get("search", legacy_search))
    feature_type = _required(features, "type", "features.type")
    feature_params = dict(features.get("params", {})) if isinstance(features.get("params", {}), dict) else {}
    feature_params.update(
        {key: value for key, value in features.items() if key not in {"type", "root", "columns", "params"}}
    )
    feature_params = _normalize_feature_params(feature_params)

    train_scheme = TrainSchemeConfig(**{**TrainSchemeConfig().__dict__, **raw.get("train_scheme", {})})
    _validate_train_scheme(train_scheme)

    return MlAlphaConfig(
        run_id=run_id,
        alpha_id=alpha_id,
        dates=DatesConfig(
            train=_required_date_pair(dates, "train"),
            valid=_optional_date_pair(dates, "valid"),
            predict=_optional_date_pair(dates, "predict"),
        ),
        sample=_sample_config(raw.get("sample", {})),
        train_scheme=train_scheme,
        label=LabelConfig(
            id=_required(label, "id", "label.id"),
            root=_project_path(label.get("root", data_root / "label" / "stock" / "daily")),
        ),
        universe=UniverseConfig(id=universe.get("id", "mkt_all")),
        filters=FiltersConfig(
            exclude_limit=bool(filters.get("exclude_limit", True)),
            exclude_st=bool(filters.get("exclude_st", True)),
            exclude_bj=bool(filters.get("exclude_bj", True)),
        ),
        preprocess=PreprocessConfig(
            cross_section_transform=preprocess.get("cross_section_transform", "none"),
            feature_fill_value=float(preprocess.get("feature_fill_value", 0.0)),
        ),
        postprocess=_postprocess_config(postprocess, data_root),
        features=FeaturesConfig(
            type=feature_type,
            root=_feature_root(features, feature_type, data_root),
            columns=_feature_columns(features.get("columns", [])),
            params=feature_params,
        ),
        materialize=MaterializeConfig(
            cache_samples=bool(materialize.get("cache_samples", False)),
            cache_dir=_project_path(materialize.get("cache_dir", data_root / "model_workspace" / run_id / "cache")),
            predict_batch_size=max(1, int(materialize.get("predict_batch_size", 20))),
        ),
        diagnostics=DiagnosticsConfig(
            enabled=bool(diagnostics.get("enabled", False)),
            print_epoch=bool(diagnostics.get("print_epoch", False)),
            write_loss_history=bool(diagnostics.get("write_loss_history", False)),
            write_model_info=bool(diagnostics.get("write_model_info", False)),
            write_window_summary=bool(diagnostics.get("write_window_summary", False)),
        ),
        model=ModelConfig(
            name=_required(model, "name", "model.name"),
            class_path=_required(model, "class", "model.class"),
            artifact_dir=_project_path(model.get("artifact_dir", data_root / "model_workspace" / run_id / "artifacts")),
            params=model_params,
            search=model_search,
        ),
        output=OutputConfig(
            kind=output_kind,
            id=output_id,
            root=output_root,
            asset=str(output.get("asset", "stock")).strip().lower(),
            frequency=str(output.get("frequency", "daily")).strip().lower(),
            base_root=_project_path(output.get("base_root", data_root / "stock_data" / "daily" / "pv")),
            write_workers=max(1, int(output.get("write_workers", 4))),
            write_metadata=bool(output.get("write_metadata", True)),
        ),
        factor_id=factor_id,
        data_root=data_root,
        output_root=output_root,
    )


def _validate_train_scheme(section: TrainSchemeConfig) -> None:
    scheme = section.type.lower()
    ratio = section.validation_ratio
    if scheme == "static" and section.train_lookback is not None:
        raise ValueError("train_scheme.train_lookback is not supported for static")
    if scheme == "static" and ratio is not None:
        raise ValueError("train_scheme.validation_ratio is not supported for static")
    if section.train_lookback is not None:
        value = section.train_lookback
        if not isinstance(value, str) or not (
            (value.lower().endswith("y") and value[:-1].isdigit())
            or (value.lower().endswith("d") and value[:-1].isdigit())
        ):
            raise ValueError("train_scheme.train_lookback must use 'Ny' or 'Nd', for example '3y' or '720d'")
    if scheme == "rolling" and ratio is not None and section.train_lookback is None:
        raise ValueError("train_scheme.validation_ratio requires train_lookback for rolling")
    if ratio is not None and not 0.0 < float(ratio) < 1.0:
        raise ValueError("train_scheme.validation_ratio must be in (0, 1)")
    if ratio is not None and scheme not in {"rolling", "expanding"}:
        raise ValueError("train_scheme.validation_ratio is only supported for rolling/expanding")
    if ratio is not None and int(section.validation_sample_count) > 0:
        raise ValueError("train_scheme.validation_ratio cannot be used with validation_sample_count")
    if ratio is not None and int(section.train_sample_count) > 0:
        raise ValueError("train_scheme.validation_ratio cannot be used with train_sample_count")
    if section.train_lookback is not None and int(section.train_sample_count) > 0:
        raise ValueError("train_scheme.train_lookback cannot be used with train_sample_count")


def _sample_config(section: dict[str, Any]) -> SampleConfig:
    train_frequency = section.get("train_frequency")
    if not train_frequency:
        raise ValueError("missing required config value: sample.train_frequency")
    return SampleConfig(
        train_frequency=str(train_frequency),
        predict_frequency=section.get("predict_frequency"),
    )


def _postprocess_config(section: dict[str, Any], data_root: Path) -> PostprocessConfig:
    _ = data_root
    legacy_keys = {"neutralize", "neutralize_rust_binary", "neutralize_temp_root"} & set(section)
    if legacy_keys:
        keys = ", ".join(sorted(f"postprocess.{key}" for key in legacy_keys))
        raise ValueError(
            f"{keys} is no longer supported; move neutralization to "
            "model.params.neutralize so the model owns the Rust import call"
        )
    return PostprocessConfig()


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _required(mapping: dict[str, Any], key: str, label: str | None = None) -> Any:
    if key not in mapping:
        raise ValueError(f"missing required config value: {label or key}")
    return mapping[key]


def _required_date_pair(section: dict[str, Any], key: str) -> DateRange:
    value = _required(section, key, f"dates.{key}")
    parsed = _parse_date_pair(value, f"dates.{key}", allow_empty=False)
    assert parsed is not None
    return parsed


def _optional_date_pair(section: dict[str, Any], key: str) -> OptionalDateRange:
    if key not in section:
        return None
    return _parse_date_pair(section[key], f"dates.{key}", allow_empty=True)


def _parse_date_pair(value: Any, label: str, allow_empty: bool) -> OptionalDateRange:
    if allow_empty and value == []:
        return None
    if not isinstance(value, list) or len(value) != 2:
        suffix = " or []" if allow_empty else ""
        raise ValueError(f"{label} must be [YYYYMMDD, YYYYMMDD]{suffix}")
    start, end = int(value[0]), int(value[1])
    if start > end:
        raise ValueError(f"{label} start must be <= end")
    return start, end


def _feature_columns(value: Any) -> list[str] | str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return [str(item) for item in value]
    raise ValueError("features.columns must be a list of column names or the string '__all__'")


def _feature_root(features: dict[str, Any], feature_type: str, data_root: Path) -> Path:
    if "root" in features:
        return _project_path(features["root"])
    if feature_type == "multi_bar_panel":
        return data_root
    raise ValueError("missing required config value: features.root")


def _normalize_feature_params(params: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(params)
    panels = normalized.get("panels")
    if isinstance(panels, dict):
        normalized_panels: dict[str, dict[str, Any]] = {}
        for name, raw_panel in panels.items():
            if not isinstance(raw_panel, dict):
                normalized_panels[name] = raw_panel
                continue
            panel = dict(raw_panel)
            if "root" in panel:
                panel["root"] = _project_path(panel["root"])
            normalized_panels[name] = panel
        normalized["panels"] = normalized_panels
    return normalized
