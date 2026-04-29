use std::env;
use std::fs;
use std::path::Path;

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set");
    let factor_root = Path::new(&manifest_dir).join("src").join("factor");
    let label_root = Path::new(&manifest_dir).join("src").join("label");
    let barra_root = Path::new(&manifest_dir).join("src").join("barra");
    println!("cargo:rerun-if-changed={}", factor_root.display());
    println!("cargo:rerun-if-changed={}", label_root.display());
    println!("cargo:rerun-if-changed={}", barra_root.display());

    let factor_module_paths = scan_registry_modules(&factor_root, "factor");
    let label_module_paths = scan_registry_modules(&label_root, "label");
    let barra_module_paths = scan_registry_modules(&barra_root, "barra");

    let out_dir = env::var("OUT_DIR").expect("OUT_DIR is set");
    fs::write(
        Path::new(&out_dir).join("factor_registry.rs"),
        generated_registry(
            "factor",
            "Factor",
            "all_factors",
            "factor_map",
            &factor_module_paths,
        ),
    )
    .expect("write generated factor registry");
    fs::write(
        Path::new(&out_dir).join("label_registry.rs"),
        generated_registry(
            "label",
            "Label",
            "all_labels",
            "label_map",
            &label_module_paths,
        ),
    )
    .expect("write generated label registry");
    fs::write(
        Path::new(&out_dir).join("barra_registry.rs"),
        generated_barra_registry(&barra_module_paths),
    )
    .expect("write generated barra registry");
}

fn scan_registry_modules(root: &Path, crate_module: &str) -> Vec<String> {
    if !root.exists() {
        return Vec::new();
    }

    let mut module_paths = Vec::new();
    for asset_entry in fs::read_dir(root).expect("read registry root") {
        let asset_entry = asset_entry.expect("read asset factor dir");
        let asset_path = asset_entry.path();
        if !asset_path.is_dir() {
            continue;
        }
        let asset_name = asset_entry.file_name().to_string_lossy().to_string();
        if asset_name == "registry" || asset_name == "common" {
            continue;
        }

        for frequency_entry in fs::read_dir(&asset_path).expect("read frequency factor dir") {
            let frequency_entry = frequency_entry.expect("read frequency entry");
            let frequency_path = frequency_entry.path();
            if !frequency_path.is_dir() {
                continue;
            }
            let frequency_name = frequency_entry.file_name().to_string_lossy().to_string();

            scan_rs_files(
                &frequency_path,
                &format!("crate::{crate_module}::{asset_name}::{frequency_name}"),
                &mut module_paths,
            );
        }
    }

    module_paths.sort();
    module_paths
}

fn scan_rs_files(path: &Path, module_prefix: &str, module_paths: &mut Vec<String>) {
    for entry in fs::read_dir(path).expect("read registry files") {
        let entry = entry.expect("read registry file");
        let entry_path = entry.path();
        let Some(stem) = entry_path.file_stem().and_then(|value| value.to_str()) else {
            continue;
        };
        if entry_path.is_dir() {
            scan_rs_files(
                &entry_path,
                &format!("{module_prefix}::{stem}"),
                module_paths,
            );
            continue;
        }
        if entry_path.extension().and_then(|value| value.to_str()) != Some("rs") || stem == "mod" {
            continue;
        }
        println!("cargo:rerun-if-changed={}", entry_path.display());
        module_paths.push(format!("{module_prefix}::{stem}"));
    }
}

fn generated_registry(
    module_name: &str,
    trait_name: &str,
    all_fn: &str,
    map_fn: &str,
    module_paths: &[String],
) -> String {
    let mut output = String::new();
    output.push_str("use std::collections::HashMap;\n\n");
    output.push_str(&format!("use crate::{module_name}::{trait_name};\n\n"));
    output.push_str(&format!(
        "pub fn {all_fn}() -> Vec<Box<dyn {trait_name}>> {{\n"
    ));
    output.push_str("    vec![\n");
    for module_path in module_paths {
        output.push_str(&format!("        {module_path}::create(),\n"));
    }
    output.push_str("    ]\n");
    output.push_str("}\n\n");
    output.push_str(&format!(
        "pub fn {map_fn}() -> HashMap<String, Box<dyn {trait_name}>> {{\n"
    ));
    output.push_str("    let mut items = HashMap::new();\n");
    output.push_str(&format!("    for item in {all_fn}() {{\n"));
    output.push_str("        let key = item.spec().registry_key();\n");
    output.push_str("        items.insert(key, item);\n");
    output.push_str("    }\n");
    output.push_str("    items\n");
    output.push_str("}\n");
    output
}

fn generated_barra_registry(module_paths: &[String]) -> String {
    let mut output = String::new();
    output.push_str("use crate::barra::BarraExposure;\n\n");
    output.push_str("pub fn all_barra_exposures() -> Vec<Box<dyn BarraExposure>> {\n");
    output.push_str("    vec![\n");
    for module_path in module_paths {
        output.push_str(&format!("        {module_path}::create(),\n"));
    }
    output.push_str("    ]\n");
    output.push_str("}\n");
    output
}
