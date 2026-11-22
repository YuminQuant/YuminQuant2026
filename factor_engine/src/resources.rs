//! Process-local compute budget, shared by provider and pure operator parallelism.
use crate::error::{err, Result};

pub fn compute_threads(requested: Option<usize>) -> Result<usize> {
    let available = std::thread::available_parallelism()?.get();
    let allocation = std::env::var("YQ_ALLOCATED_CPUS").ok()
        .map(|value| value.parse::<usize>()).transpose()?.unwrap_or(available);
    let limit = available.min(allocation);
    let configured = std::env::var("YQ_COMPUTE_THREADS").ok()
        .map(|value| value.parse::<usize>()).transpose()?;
    let threads = requested.or(configured).unwrap_or_else(|| limit.max(1));
    if threads == 0 || threads > limit {
        return Err(err(format!("threads must be in 1..={limit} (effective CPU allocation), got {threads}")));
    }
    Ok(threads)
}

pub fn thread_pool(requested: Option<usize>) -> Result<rayon::ThreadPool> {
    Ok(rayon::ThreadPoolBuilder::new().num_threads(compute_threads(requested)?).build()?)
}

pub(crate) fn parallel_work(len: usize) -> bool {
    len >= 65_536 && rayon::current_thread_index().is_some() && rayon::current_num_threads() > 1
}

pub(crate) fn stock_wave_width(_dates: usize, _inputs: usize) -> usize {
    rayon::current_num_threads().max(1)
}
