//! Ordered, memory-budgeted projected reads; local inputs stay serial.
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Condvar, Mutex, OnceLock};
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
    budget: usize,
    spacing: Duration,
    next_open: Mutex<Instant>,
}
static REMOTE: OnceLock<ReadAhead> = OnceLock::new();
static INIT: Mutex<()> = Mutex::new(());
static THREADS: AtomicUsize = AtomicUsize::new(0);
static PEAK_RESERVED: AtomicUsize = AtomicUsize::new(0);
static OVERSIZED: AtomicUsize = AtomicUsize::new(0);
static ESTIMATE_EXCEEDED: AtomicUsize = AtomicUsize::new(0);

pub(crate) fn configure_threads(threads: usize) { THREADS.store(threads, Ordering::Relaxed); }

pub(crate) fn profile() -> String {
    format!("prefetch_reserved_peak={} prefetch_oversized={} prefetch_estimate_exceeded={}",
        PEAK_RESERVED.load(Ordering::Relaxed), OVERSIZED.load(Ordering::Relaxed),
        ESTIMATE_EXCEEDED.load(Ordering::Relaxed))
}

impl ReadAhead {
    fn new(options: ReadOptions, adapter_paces: bool) -> Result<Self> {
        if options.concurrency == 0 || !options.opens_per_second.is_finite()
            || (options.opens_per_second != 0.0 && !(0.01..=10000.0).contains(&options.opens_per_second))
            || options.prefetch_bytes < crate::storage_io::READER_CACHE_BYTES
        { return Err(err("invalid remote read options")); }
        let requested = THREADS.load(Ordering::Relaxed);
        let width = options.concurrency.min(options.prefetch_bytes / crate::storage_io::READER_CACHE_BYTES);
        Ok(Self {
            io: rayon::ThreadPoolBuilder::new().num_threads(width).thread_name(|i| format!("input-plan-{i}")).build()?,
            compute: crate::resources::thread_pool((requested > 0).then_some(requested))?,
            width,
            budget: options.prefetch_bytes,
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

#[derive(Default)]
struct BudgetState { used: usize, next: usize, cancelled: bool }
struct Budget { state: Mutex<BudgetState>, changed: Condvar, limit: usize }
struct Reservation { budget: Arc<Budget>, bytes: usize }
impl Budget {
    fn acquire(self: &Arc<Self>, index: usize, bytes: usize) -> Option<Reservation> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        // Admit in file order: later results cannot consume the budget needed by
        // the earliest result, which the ordered consumer must receive first.
        while !state.cancelled && (state.next != index ||
            (state.used != 0 && bytes > self.limit.saturating_sub(state.used)))
        { state = self.changed.wait(state).unwrap_or_else(|e| e.into_inner()); }
        if state.cancelled { return None; }
        if bytes > self.limit { OVERSIZED.fetch_add(1, Ordering::Relaxed); }
        state.used = state.used.saturating_add(bytes);
        state.next += 1;
        PEAK_RESERVED.fetch_max(state.used, Ordering::Relaxed);
        self.changed.notify_all();
        Some(Reservation { budget: Arc::clone(self), bytes })
    }

    fn cancel(&self) {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).cancelled = true;
        self.changed.notify_all();
    }
}
impl Drop for Reservation {
    fn drop(&mut self) {
        let mut state = self.budget.state.lock().unwrap_or_else(|e| e.into_inner());
        state.used = state.used.saturating_sub(self.bytes);
        self.budget.changed.notify_all();
    }
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
/// completed tables and their reservations remain bounded until ordered consumption.
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
    let budget = Arc::new(Budget { state: Mutex::new(BudgetState::default()),
        changed: Condvar::new(), limit: reader.budget });
    let next = AtomicUsize::new(0);
    let (tx, rx) = mpsc::channel();
    std::thread::scope(|scope| {
        for _ in 0..reader.width.min(paths.len()) {
            let (tx, budget, next, prepare, finish) = (tx.clone(), Arc::clone(&budget), &next, &prepare, &finish);
            scope.spawn(move || loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                if index >= paths.len() { break; }
                let path = &paths[index];
                let prepared = std::panic::catch_unwind(std::panic::AssertUnwindSafe(||
                    prepare(index, reader.open(path)?)))
                    .unwrap_or_else(|_| Err(err(format!("projected read preparation panicked: {}", path.display()))));
                let bytes = prepared.as_ref().map_or(0, |p| p.reserved_bytes);
                let Some(reservation) = budget.acquire(index, bytes) else { break; };
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let prepared = prepared?;
                    prepared.prefetch()?;
                    let table = reader.compute.install(|| finish(index, prepared.read(path)?))?;
                    if table.allocated_bytes() > bytes { ESTIMATE_EXCEEDED.fetch_add(1, Ordering::Relaxed); }
                    Ok(table)
                })).unwrap_or_else(|_| Err(err(format!("projected read panicked: {}", path.display()))));
                if tx.send((index, result, reservation)).is_err() { break; }
            });
        }
        drop(tx);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut pending = BTreeMap::new();
            for index in 0..paths.len() {
                while !pending.contains_key(&index) {
                    let (ready, table, reservation) = rx.recv().map_err(|_| err("projected read workers stopped early"))?;
                    pending.insert(ready, (table, reservation));
                }
                let (table, reservation) = pending.remove(&index).expect("ordered result received");
                consume(index, table?)?;
                drop(reservation);
            }
            Ok(())
        })).unwrap_or_else(|_| Err(err("projected read consumer panicked")));
        budget.cancel();
        drop(rx);
        result
    })
}
