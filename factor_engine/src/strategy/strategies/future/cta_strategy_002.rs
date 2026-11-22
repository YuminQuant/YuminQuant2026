//! P99/P1 entries, P5/P95 exits or a 60-trading-minute holding timeout.
use super::cta_strategy_001::CtaStrategy001;
use crate::error::Result;
use crate::strategy::config::StrategyRunConfig;

pub fn create(config: &StrategyRunConfig) -> Result<CtaStrategy001> {
    CtaStrategy001::from_config(config)
}
