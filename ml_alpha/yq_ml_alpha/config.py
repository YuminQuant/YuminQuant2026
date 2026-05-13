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
    rolling_train_days: int = 756
    valid_days: int = 252
    train_sample_count: int = 0
    validation_sample_count: int = 0


@dataclass(frozen=True)
class PreprocessConfig:
    cross_section_transform: str = "none"
    feature_fill_value: float = 0.0


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
    features: FeaturesConfig
    materialize: MaterializeConfig
    diagnostics: DiagnosticsConfig
    model: ModelConfig
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
    output_root = _project_path(raw.get("output_root", "data/models"))
    run_id = _required(raw, "run_id")
    alpha_id = _required(raw, "alpha_id")

    dates = raw.get("dates", {})
    label = raw.get("label", {})
    universe = raw.get("universe", {})
    filters = raw.get("filters", {})
    preprocess = raw.get("preprocess", {})
    features = raw.get("features", {})
    materialize = raw.get("materialize", {})
    diagnostics = raw.get("diagnostics", {})
    model = raw.get("model", {})

    model_params = dict(model.get("params", {}))
    legacy_search = model_params.pop("search", {})
    model_search = dict(model.get("search", legacy_search))

    return MlAlphaConfig(
        run_id=run_id,
        alpha_id=alpha_id,
        dates=DatesConfig(
            train=_required_date_pair(dates, "train"),
            valid=_optional_date_pair(dates, "valid"),
            predict=_optional_date_pair(dates, "predict"),
        ),
        sample=_sample_config(raw.get("sample", {})),
        train_scheme=TrainSchemeConfig(**{**TrainSchemeConfig().__dict__, **raw.get("train_scheme", {})}),
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
        features=FeaturesConfig(
            type=_required(features, "type", "features.type"),
            root=_project_path(_required(features, "root", "features.root")),
            columns=_feature_columns(features.get("columns", [])),
            params={key: value for key, value in features.items() if key not in {"type", "root", "columns"}},
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
        data_root=data_root,
        output_root=output_root,
    )


def _sample_config(section: dict[str, Any]) -> SampleConfig:
    train_frequency = section.get("train_frequency")
    if not train_frequency:
        raise ValueError("missing required config value: sample.train_frequency")
    return SampleConfig(
        train_frequency=str(train_frequency),
        predict_frequency=section.get("predict_frequency"),
    )


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
