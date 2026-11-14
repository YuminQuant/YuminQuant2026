use crate::error::Result;
use crate::strategy::context::StrategyContext;
use crate::strategy::market::{BarEvent, SessionOpenEvent};
use crate::strategy::order::FillEvent;
use crate::strategy::session_io::{SessionFile, SessionInputs};

pub trait Strategy {
    fn name(&self) -> &'static str;

    fn prepare_run(
        &mut self,
        _config: &crate::strategy::config::StrategyRunConfig,
        _dates: &[i32],
    ) -> Result<()> {
        Ok(())
    }

    fn prepare_future_run(
        &mut self,
        _config: &crate::strategy::config::StrategyRunConfig,
        _calendar: &crate::calendar::TradingCalendar,
        _dates: &[i32],
    ) -> Result<()> {
        Ok(())
    }

    /// All listed contracts are available to the strategy before the minute stream is loaded.
    fn select_future_contracts(
        &mut self,
        _config: &crate::strategy::config::StrategyRunConfig,
        _date: i32,
        _contracts: &std::collections::BTreeMap<
            String,
            crate::strategy::market::FutureContractMeta,
        >,
    ) -> Result<Option<std::collections::BTreeSet<String>>> {
        Ok(None)
    }

    /// Input batches must not cross a strategy decision boundary.
    fn input_batch_days(&self) -> usize {
        1
    }

    /// Declare independent files after contract selection and before daily input IO.
    fn session_files(
        &self,
        _config: &crate::strategy::config::StrategyRunConfig,
        _date: i32,
        _symbols: Option<&std::collections::BTreeSet<String>>,
    ) -> Vec<SessionFile> {
        Vec::new()
    }

    /// Called on the replay thread with the inputs for this date.
    fn on_session_inputs(&mut self, _date: i32, _inputs: SessionInputs) -> Result<()> {
        Ok(())
    }

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

    fn on_session_end(&mut self, _ctx: &mut StrategyContext, _event: &BarEvent) -> Result<()> {
        Ok(())
    }

    fn on_end(&mut self, _ctx: &mut StrategyContext) -> Result<()> {
        Ok(())
    }
}
