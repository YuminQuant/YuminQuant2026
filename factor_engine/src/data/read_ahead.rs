//! Bounded remote open/footer prefetch; projected tables are decoded in input order.
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use parquet::file::reader::{ChunkReader, Length};
use rayon::prelude::*;

use crate::artifact_io::{self, ReadOptions, StorageReader};
use crate::error::{err, Result};

struct ReadAhead {
    pool: rayon::ThreadPool,
    width: usize,
    spacing: Duration,
    next_open: Mutex<Instant>,
}
static REMOTE: OnceLock<ReadAhead> = OnceLock::new();
static INIT: Mutex<()> = Mutex::new(());

impl ReadAhead {
    fn new(options: ReadOptions) -> Result<Self> {
        if options.concurrency == 0 || !options.opens_per_second.is_finite()
            || (options.opens_per_second != 0.0 && !(0.01..=10000.0).contains(&options.opens_per_second))
            || options.prefetch_bytes < crate::storage_io::READER_CACHE_BYTES
        {
            return Err(err("invalid remote read options: concurrency > 0, rate 0 or 0.01..=10000, prefetch >= 1 MiB required"));
        }
        let width = options.concurrency.min(options.prefetch_bytes / crate::storage_io::READER_CACHE_BYTES);
        Ok(Self {
            pool: rayon::ThreadPoolBuilder::new().num_threads(width).thread_name(|i| format!("input-io-{i}")).build()?,
            width,
            spacing: if options.opens_per_second == 0.0 { Duration::ZERO } else { Duration::from_secs_f64(1.0 / options.opens_per_second) },
            next_open: Mutex::new(Instant::now()),
        })
    }

    fn open(&self, path: &Path) -> Result<StorageReader> {
        let start = {
            let mut next = self.next_open.lock().map_err(|_| err("read-ahead rate limiter poisoned"))?;
            let start = (*next).max(Instant::now());
            *next = start + self.spacing;
            start
        };
        std::thread::sleep(start.saturating_duration_since(Instant::now()));
        let reader = artifact_io::open_reader(path)?;
        // Warm at most one block, not a whole file or a decoded daily table.
        let start = reader.len().saturating_sub(1) / (64 * 1024) * (64 * 1024);
        reader.get_bytes(start, (reader.len() - start) as usize)?;
        Ok(reader)
    }
}

/// A wave owns at most `width` handles and `width * 1 MiB` range-cache bytes.
/// Decode/consume one file before advancing; errors drop every unopened result.
pub(crate) fn visit_files(
    paths: &[PathBuf],
    mut consume: impl FnMut(&Path, StorageReader) -> Result<()>,
) -> Result<()> {
    if paths.is_empty() { return Ok(()); }
    let Some(backend) = artifact_io::backend(&paths[0]) else {
        for path in paths { consume(path, artifact_io::open_reader(path)?)?; }
        return Ok(());
    };
    if REMOTE.get().is_none() {
        let _init = INIT.lock().map_err(|_| err("read-ahead initialization poisoned"))?;
        if REMOTE.get().is_none() {
            let _ = REMOTE.set(ReadAhead::new(backend.read_options())?);
        }
    }
    let read_ahead = REMOTE.get().expect("initialized remote reader");
    for wave in paths.chunks(read_ahead.width) {
        let readers = read_ahead.pool.install(|| wave.par_iter().map(|path| read_ahead.open(path)).collect::<Vec<_>>());
        for (path, reader) in wave.iter().zip(readers) { consume(path, reader?)?; }
    }
    Ok(())
}
