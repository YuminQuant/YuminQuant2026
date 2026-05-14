use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct StrategyRunRequest {
    pub config_path: PathBuf,
    pub project_config_path: Option<PathBuf>,
    pub detail: Option<bool>,
}
