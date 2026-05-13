use crate::error::Result;
use crate::strategy::context::StrategyContext;
use crate::strategy::market::{BarEvent, SessionOpenEvent};
use crate::strategy::order::FillEvent;

pub trait Strategy {
    fn name(&self) -> &'static str;

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
