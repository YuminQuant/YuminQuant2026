//! Optional output storage supplied by an external launcher. Local IO is the default.
use std::path::Path;
use std::sync::{Arc, OnceLock};
use parquet::basic::Compression;
use crate::data::table::Table;
use crate::error::{err, Result};

pub trait ArtifactGuard {}
impl<T> ArtifactGuard for T {}

pub trait ArtifactBackend: Send + Sync {
    fn handles(&self, path: &Path) -> bool;
    fn exists(&self, path: &Path) -> Result<bool>;
    fn create_dir_all(&self, path: &Path) -> Result<()>;
    fn read_table(&self, path: &Path, columns: Option<&[String]>) -> Result<Table>;
    fn write_table(&self, path: &Path, table: &Table, compression: Compression) -> Result<()>;
    fn lock(&self, path: &Path) -> Result<Box<dyn ArtifactGuard>>;
}

static BACKEND: OnceLock<Arc<dyn ArtifactBackend>> = OnceLock::new();

pub fn register(backend: Arc<dyn ArtifactBackend>) -> Result<()> {
    BACKEND.set(backend).map_err(|_| err("artifact backend already registered"))
}

pub fn backend(path: &Path) -> Option<&'static dyn ArtifactBackend> {
    BACKEND.get().map(|b| b.as_ref()).filter(|b| b.handles(path))
}

pub fn exists(path: &Path) -> Result<bool> {
    match backend(path) { Some(b) => b.exists(path), None => Ok(path.exists()) }
}

pub fn create_dir_all(path: &Path) -> Result<()> {
    match backend(path) { Some(b) => b.create_dir_all(path), None => Ok(std::fs::create_dir_all(path)?) }
}

pub fn lock(path: &Path) -> Result<Option<Box<dyn ArtifactGuard>>> {
    backend(path).map(|b| b.lock(path)).transpose()
}
