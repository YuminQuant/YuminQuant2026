//! Ordered projected reads; local inputs stay serial.
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Mutex, OnceLock};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use rayon::prelude::*;

use crate::artifact_io::{self, ReadOptions, StorageReader};
use crate::data::parquet_io::PreparedProjection;
use crate::data::Table;
use crate::error::{err, Result};

struct ReadAhead {
    io: rayon::ThreadPool,
    compute: rayon::ThreadPool,
    width: usize,
    spacing: Duration,
    next_open: Mutex<Instant>,
}
static REMOTE: OnceLock<ReadAhead> = OnceLock::new();
static INIT: Mutex<()> = Mutex::new(());
static THREADS: AtomicUsize = AtomicUsize::new(0);
pub(crate) fn configure_threads(threads: usize) { THREADS.store(threads, Ordering::Relaxed); }

impl ReadAhead {
    fn new(options: ReadOptions, adapter_paces: bool) -> Result<Self> {
        if options.concurrency == 0 || !options.opens_per_second.is_finite()
            || (options.opens_per_second != 0.0 && !(0.01..=10000.0).contains(&options.opens_per_second))
        { return Err(err("invalid remote read options")); }
        let requested = THREADS.load(Ordering::Relaxed);
        let width = options.concurrency;
        Ok(Self {
            io: rayon::ThreadPoolBuilder::new().num_threads(width).thread_name(|i| format!("input-plan-{i}")).build()?,
            compute: crate::resources::thread_pool((requested > 0).then_some(requested))?,
            width,
            spacing: if adapter_paces || options.opens_per_second == 0.0 { Duration::ZERO }
                else { Duration::from_secs_f64(1.0 / options.opens_per_second) },
            next_open: Mutex::new(Instant::now()),
        })
    }

    fn open(&self, path: &Path) -> Result<StorageReader> {
        if !self.spacing.is_zero() {
            let start = {
                let mut next = self.next_open.lock().map_err(|_| err("read rate limiter poisoned"))?;
                let start = (*next).max(Instant::now());
                *next = start + self.spacing;
                start
            };
            std::thread::sleep(start.saturating_duration_since(Instant::now()));
        }
        artifact_io::open_reader(path)
    }
}

fn remote(path: &Path) -> Result<Option<&'static ReadAhead>> {
    let Some(backend) = artifact_io::backend(path) else { return Ok(None); };
    if REMOTE.get().is_none() {
        let _init = INIT.lock().map_err(|_| err("read-ahead initialization poisoned"))?;
        if REMOTE.get().is_none() {
            let _ = REMOTE.set(ReadAhead::new(backend.read_options(), backend.paces_read_opens())?);
        }
    }
    Ok(REMOTE.get())
}

/// Small file plans/status results only, never decoded historical tables.
pub(crate) fn plan_files<T: Send>(paths: &[PathBuf], plan: impl Fn(&Path) -> Result<T> + Sync + Send) -> Result<Vec<T>> {
    if let Some(first) = paths.first() {
        if let Some(reader) = remote(first)? {
            return reader.io.install(|| paths.par_iter().map(|path| plan(path)).collect());
        }
    }
    paths.iter().map(|path| plan(path)).collect()
}

pub(crate) fn visit_projected_files(
    paths: &[PathBuf], columns: &[String],
    mut consume: impl FnMut(&Path, Table) -> Result<()>,
) -> Result<()> {
    visit_tables(paths,
        |i, reader| PreparedProjection::new(reader, &paths[i], columns, &BTreeSet::new()),
        |_, table| Ok(table),
        |i, table| consume(&paths[i], table))
}

/// IO workers fetch selected compressed chunks. The compute pool decodes them;
/// each worker hands off results in file order before preparing its next file.
pub(crate) fn visit_tables(
    paths: &[PathBuf],
    prepare: impl Fn(usize, StorageReader) -> Result<PreparedProjection> + Sync,
    finish: impl Fn(usize, Table) -> Result<Table> + Sync,
    mut consume: impl FnMut(usize, Table) -> Result<()>,
) -> Result<()> {
    if paths.is_empty() { return Ok(()); }
    let Some(reader) = remote(&paths[0])? else {
        for (i, path) in paths.iter().enumerate() {
            let prepared = prepare(i, artifact_io::open_reader(path)?)?;
            consume(i, finish(i, prepared.read(path)?)?)?;
        }
        return Ok(());
    };
    let width = reader.width.min(paths.len());
    std::thread::scope(|scope| {
        let mut receivers = Vec::with_capacity(width);
        for lane in 0..width {
            let (tx, rx) = mpsc::sync_channel(0);
            receivers.push(rx);
            let (prepare, finish) = (&prepare, &finish);
            scope.spawn(move || {
                for index in (lane..paths.len()).step_by(width) {
                    let path = &paths[index];
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let prepared = prepare(index, reader.open(path)?)?;
                        prepared.prefetch()?;
                        reader.compute.install(|| finish(index, prepared.read(path)?))
                    }))
                    .unwrap_or_else(|_| Err(err(format!("projected read panicked: {}", path.display()))));
                    if tx.send(result).is_err() {
                        break;
                    }
                }
            });
        }
        // Dropping the receivers on error also releases workers waiting to hand off a table.
        for index in 0..paths.len() {
            let table = receivers[index % width].recv()
                .map_err(|_| err("projected read workers stopped early"))??;
            consume(index, table)?;
        }
        Ok(())
    })
}
