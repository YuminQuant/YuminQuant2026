use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use rayon::prelude::*;

use crate::artifact_io::StoragePathExt;
use crate::data::parquet_io::{read_parquet, read_parquet_symbols};
use crate::data::Table;
use crate::error::{err, Result};

static POOL: OnceLock<std::result::Result<rayon::ThreadPool, rayon::ThreadPoolBuildError>> =
    OnceLock::new();

fn pool() -> Result<&'static rayon::ThreadPool> {
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(50)
            .thread_name(|index| format!("strategy-input-{index}"))
            .build()
    })
    .as_ref()
    .map_err(|error| err(format!("cannot start strategy IO pool: {error}")))
}

/// Run independent input tasks on the shared pool. Consumers apply results in order.
pub(crate) fn join<A: Send, B: Send>(
    left: impl FnOnce() -> A + Send,
    right: impl FnOnce() -> B + Send,
) -> Result<(A, B)> {
    Ok(pool()?.install(|| rayon::join(left, right)))
}

pub(crate) fn map<I: Send, T: Send>(
    items: Vec<I>,
    read: impl Fn(I) -> Result<T> + Send + Sync,
) -> Result<Vec<T>> {
    if items.len() < 2 {
        return items.into_iter().map(read).collect();
    }
    pool()?
        .install(|| items.into_par_iter().map(read).collect::<Vec<_>>())
        .into_iter()
        .collect()
}

/// Additional input declared by a strategy for the current trade date.
pub struct SessionFile {
    pub name: String,
    pub path: PathBuf,
    pub columns: Vec<String>,
    pub symbols: Option<BTreeSet<String>>,
    pub optional: bool,
}

impl SessionFile {
    pub(crate) fn load(self) -> Result<(String, Option<Table>)> {
        if self.optional && !self.path.storage_exists() {
            return Ok((self.name, None));
        }
        let columns = (!self.columns.is_empty()).then_some(self.columns.as_slice());
        let table = match self.symbols {
            Some(symbols) => read_parquet_symbols(&self.path, &self.columns, move |code| {
                symbols.contains(code)
            })?,
            None => read_parquet(&self.path, columns)?,
        };
        Ok((self.name, Some(table)))
    }
}

pub type SessionInputs = BTreeMap<String, Option<Table>>;

pub(crate) fn load_files(files: Vec<SessionFile>) -> Result<SessionInputs> {
    let mut inputs = SessionInputs::new();
    for (name, table) in map(files, SessionFile::load)? {
        if inputs.insert(name.clone(), table).is_some() {
            return Err(err(format!("duplicate strategy session input {name}")));
        }
    }
    Ok(inputs)
}

#[derive(Clone, Debug)]
pub(crate) struct InputTemplate {
    pub path: PathBuf,
    pub columns: Vec<String>,
    pub optional: bool,
}

impl InputTemplate {
    pub fn for_date(&self, root: &Path, name: &str, date: i32) -> SessionFile {
        let path = self
            .path
            .to_string_lossy()
            .replace("{date}", &date.to_string())
            .replace("{year}", &(date / 10000).to_string());
        SessionFile {
            name: name.to_owned(),
            path: root.join(path),
            columns: self.columns.clone(),
            symbols: None,
            optional: self.optional,
        }
    }
}
