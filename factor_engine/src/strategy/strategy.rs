use crate::error::Result;
use crate::strategy::context::StrategyContext;
use crate::strategy::market::{BarEvent, SessionOpenEvent};
use crate::strategy::order::FillEvent;

pub trait Strategy {
    fn name(&self) -> &'static str;

    /// Prepare today's universe before market data is loaded. Never uses today's signal.
    fn prepare_session(
        &mut self,
        _config: &crate::strategy::config::StrategyRunConfig,
        _account: &crate::strategy::account::AccountState,
        _date: i32,
    ) -> Result<Option<std::collections::BTreeSet<String>>> {
        Ok(None)
    }

    /// Orders execute at this bar's open; implementations must only inspect open-time information.
    fn on_bar_open(&mut self, _ctx: &mut StrategyContext, _event: &BarEvent) -> Result<()> {
        Ok(())
    }

    fn on_start(&mut self, _ctx: &mut StrategyContext) -> Result<()> {
        Ok(())
    }

    fn on_session_open(
        &mut self,
        _ctx: &mut StrategyContext,
        _event: &SessionOpenEvent,
    ) -> Result<()> {
        Ok(())
    }

    fn on_bar(&mut self, ctx: &mut StrategyContext, event: &BarEvent) -> Result<()>;

    fn on_fill(&mut self, _ctx: &mut StrategyContext, _event: &FillEvent) -> Result<()> {
        Ok(())
    }

    fn on_end(&mut self, _ctx: &mut StrategyContext) -> Result<()> {
        Ok(())
    }
}
