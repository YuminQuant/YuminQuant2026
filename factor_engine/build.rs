use std::env;
use std::fs;
use std::path::Path;

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set");
    let factor_root = Path::new(&manifest_dir).join("src").join("factor");
    println!("cargo:rerun-if-changed={}", factor_root.display());

    let mut module_paths = Vec::new();
    for asset_entry in fs::read_dir(&factor_root).expect("read factor root") {
        let asset_entry = asset_entry.expect("read asset factor dir");
        let asset_path = asset_entry.path();
        if !asset_path.is_dir() {
            continue;
        }
        let asset_name = asset_entry.file_name().to_string_lossy().to_string();
        if asset_name == "registry" {
            continue;
        }

        for frequency_entry in fs::read_dir(&asset_path).expect("read frequency factor dir") {
            let frequency_entry = frequency_entry.expect("read frequency entry");
            let frequency_path = frequency_entry.path();
            if !frequency_path.is_dir() {
                continue;
            }
            let frequency_name = frequency_entry.file_name().to_string_lossy().to_string();

            for factor_entry in fs::read_dir(&frequency_path).expect("read factor files") {
                let factor_entry = factor_entry.expect("read factor file");
                let factor_path = factor_entry.path();
                if factor_path.extension().and_then(|value| value.to_str()) != Some("rs") {
                    continue;
                }
                let Some(stem) = factor_path.file_stem().and_then(|value| value.to_str()) else {
                    continue;
                };
                if stem == "mod" {
                    continue;
                }
                println!("cargo:rerun-if-changed={}", factor_path.display());
                module_paths.push(format!(
                    "crate::factor::{asset_name}::{frequency_name}::{stem}"
                ));
            }
        }
    }

    module_paths.sort();

    let mut output = String::new();
    output.push_str("use std::collections::HashMap;\n\n");
    output.push_str("use crate::factor::Factor;\n\n");
    output.push_str("pub fn all_factors() -> Vec<Box<dyn Factor>> {\n");
    output.push_str("    vec![\n");
    for module_path in &module_paths {
        output.push_str(&format!("        {module_path}::create(),\n"));
    }
    output.push_str("    ]\n");
    output.push_str("}\n\n");
    output.push_str("pub fn factor_map() -> HashMap<String, Box<dyn Factor>> {\n");
    output.push_str("    let mut factors = HashMap::new();\n");
    output.push_str("    for factor in all_factors() {\n");
    output.push_str("        let key = factor.spec().registry_key();\n");
    output.push_str("        factors.insert(key, factor);\n");
    output.push_str("    }\n");
    output.push_str("    factors\n");
    output.push_str("}\n");

    let out_dir = env::var("OUT_DIR").expect("OUT_DIR is set");
    fs::write(Path::new(&out_dir).join("factor_registry.rs"), output)
        .expect("write generated factor registry");
}
