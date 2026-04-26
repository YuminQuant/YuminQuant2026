import argparse
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Create a new Rust factor skeleton.")
    parser.add_argument("--asset", choices=["stock", "chn_stock", "future"], required=True)
    parser.add_argument("--frequency", choices=["daily", "minute", "minute_1m"], required=True)
    parser.add_argument("--name", required=True, help="snake_case factor id and file name.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing factor file.")
    return parser.parse_args()


def main():
    args = parse_args()
    name = normalize_name(args.name)
    asset_dir = "chn_stock" if args.asset in {"stock", "chn_stock"} else "future"
    frequency_dir = "minute" if args.frequency in {"minute", "minute_1m"} else "daily"
    factor_engine_dir = Path(__file__).resolve().parents[1]
    target_dir = factor_engine_dir / "src" / "factor" / asset_dir / frequency_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / f"{name}.rs"
    if target_file.exists() and not args.force:
        raise FileExistsError(f"factor already exists: {target_file}")

    struct_name = struct_name_for(asset_dir, frequency_dir, name)
    target_file.write_text(template(asset_dir, frequency_dir, name, struct_name), encoding="utf-8")
    update_mod_rs(target_dir / "mod.rs", name, struct_name)
    print(f"created {target_file}")
    print(f"updated {target_dir / 'mod.rs'}")


def normalize_name(value):
    if not re.fullmatch(r"[a-z][a-z0-9_]*", value):
        raise ValueError("--name must be snake_case and start with a lowercase letter")
    return value


def struct_name_for(asset_dir, frequency_dir, name):
    prefix = "Stock" if asset_dir == "chn_stock" else "Future"
    frequency = "Daily" if frequency_dir == "daily" else "Minute"
    body = "".join(part.capitalize() for part in name.split("_"))
    return f"{prefix}{frequency}{body}"


def template(asset_dir, frequency_dir, name, struct_name):
    asset_class = "Stock" if asset_dir == "chn_stock" else "Future"
    frequency = "Daily" if frequency_dir == "daily" else "Minute1"
    return f"""use crate::core::{{
    AssetClass, FactorContext, FactorSeries, FactorSpec, Frequency, Lookback,
}};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::Factor;

pub struct {struct_name};

pub fn create() -> Box<dyn Factor> {{
    Box::new({struct_name})
}}

impl Factor for {struct_name} {{
    fn spec(&self) -> FactorSpec {{
        FactorSpec {{
            id: \"{name}\".to_string(),
            aliases: Vec::new(),
            name: \"{name}\".to_string(),
            asset_class: AssetClass::{asset_class},
            frequency: Frequency::{frequency},
            version: \"0.1.0\".to_string(),
            tags: [\"TODO\"].iter().map(|value| value.to_string()).collect(),
            description: \"TODO\".to_string(),
            dependencies: Vec::new(),
            lookback: Lookback {{ trading_days: 0 }},
        }}
    }}

    fn compute(&self, _context: &FactorContext, _data: &DataPool) -> Result<FactorSeries> {{
        todo!(\"implement {name}\")
    }}
}}
"""


def update_mod_rs(path, module_name, struct_name):
    content = path.read_text(encoding="utf-8") if path.exists() else ""
    mod_line = f"pub mod {module_name};"
    use_line = f"pub use {module_name}::{struct_name};"
    lines = content.splitlines()
    if mod_line not in lines:
        insert_at = next((idx for idx, line in enumerate(lines) if not line.startswith("pub mod ")), len(lines))
        lines.insert(insert_at, mod_line)
    if use_line not in lines:
        lines.append(use_line)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
