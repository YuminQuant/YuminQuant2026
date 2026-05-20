use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct AnalystConsensusRequest {
    pub start_date: i32,
    pub end_date: i32,
    pub output_root: PathBuf,
}

pub const ANALYST_CONSENSUS_RELATIVE_ROOT: &str = "derived/stock/analyst_consensus";
