from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


DateRange = tuple[int, int]


@dataclass(frozen=True)
class DatesConfig:
    train: DateRange
    valid: DateRange
    predict: DateRange


@dataclass(frozen=True)
class SampleConfig:
    frequency: str = "monthly_end"


@dataclass(frozen=True)
class TrainSchemeConfig:
    type: str = "static"
    refit_frequency: str = "monthly"
    min_train_days: int = 756
    rolling_train_days: int = 756
    valid_days: int = 252


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
    columns: list[str]
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MaterializeConfig:
    cache_samples: bool = False
    cache_dir: Path = Path("data/model_workspace/cache")


@dataclass(frozen=True)
class ModelConfig:
    name: str
    class_path: str
    artifact_dir: Path
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TuningConfig:
    enabled: bool = False
    method: str = "optuna"
    params: dict[str, Any] = field(default_factory=dict)


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
    features: FeaturesConfig
    materialize: MaterializeConfig
    model: ModelConfig
    tuning: TuningConfig
    data_root: Path = Path("data")
    output_root: Path = Path("data/models")


def load_config(path: str | Path) -> MlAlphaConfig:
    config_path = Path(path)
    with config_path.open("rb") as file:
        raw = tomllib.load(file)

    data_root = Path(raw.get("data_root", "data"))
    output_root = Path(raw.get("output_root", "data/models"))
    run_id = _required(raw, "run_id")
    alpha_id = _required(raw, "alpha_id")

    dates = raw.get("dates", {})
    label = raw.get("label", {})
    universe = raw.get("universe", {})
    filters = raw.get("filters", {})
    features = raw.get("features", {})
    materialize = raw.get("materialize", {})
    model = raw.get("model", {})
    tuning = raw.get("tuning", {})

    return MlAlphaConfig(
        run_id=run_id,
        alpha_id=alpha_id,
        dates=DatesConfig(
            train=_date_pair(dates, "train"),
            valid=_date_pair(dates, "valid"),
            predict=_date_pair(dates, "predict"),
        ),
        sample=SampleConfig(frequency=raw.get("sample", {}).get("frequency", "monthly_end")),
        train_scheme=TrainSchemeConfig(**{**TrainSchemeConfig().__dict__, **raw.get("train_scheme", {})}),
        label=LabelConfig(
            id=_required(label, "id", "label.id"),
            root=Path(label.get("root", data_root / "label" / "stock" / "daily")),
        ),
        universe=UniverseConfig(id=universe.get("id", "mkt_all")),
        filters=FiltersConfig(
            exclude_limit=bool(filters.get("exclude_limit", True)),
            exclude_st=bool(filters.get("exclude_st", True)),
            exclude_bj=bool(filters.get("exclude_bj", True)),
        ),
        features=FeaturesConfig(
            type=_required(features, "type", "features.type"),
            root=Path(_required(features, "root", "features.root")),
            columns=list(features.get("columns", [])),
            params={key: value for key, value in features.items() if key not in {"type", "root", "columns"}},
        ),
        materialize=MaterializeConfig(
            cache_samples=bool(materialize.get("cache_samples", False)),
            cache_dir=Path(materialize.get("cache_dir", data_root / "model_workspace" / run_id / "cache")),
        ),
        model=ModelConfig(
            name=_required(model, "name", "model.name"),
            class_path=_required(model, "class", "model.class"),
            artifact_dir=Path(model.get("artifact_dir", data_root / "model_workspace" / run_id / "artifacts")),
            params=dict(model.get("params", {})),
        ),
        tuning=TuningConfig(
            enabled=bool(tuning.get("enabled", False)),
            method=tuning.get("method", "optuna"),
            params=_nested_params(tuning, {"enabled", "method"}),
        ),
        data_root=data_root,
        output_root=output_root,
    )


def _required(mapping: dict[str, Any], key: str, label: str | None = None) -> Any:
    if key not in mapping:
        raise ValueError(f"missing required config value: {label or key}")
    return mapping[key]


def _date_pair(section: dict[str, Any], key: str) -> DateRange:
    value = _required(section, key, f"dates.{key}")
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"dates.{key} must be [YYYYMMDD, YYYYMMDD]")
    start, end = int(value[0]), int(value[1])
    if start > end:
        raise ValueError(f"dates.{key} start must be <= end")
    return start, end


def _nested_params(section: dict[str, Any], excluded: set[str]) -> dict[str, Any]:
    if isinstance(section.get("params"), dict):
        return dict(section["params"])
    return {key: value for key, value in section.items() if key not in excluded}
