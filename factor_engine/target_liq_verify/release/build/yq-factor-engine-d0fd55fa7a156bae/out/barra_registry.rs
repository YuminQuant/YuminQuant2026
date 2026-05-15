use crate::barra::BarraExposure;

pub fn all_barra_exposures() -> Vec<Box<dyn BarraExposure>> {
    vec![
        crate::barra::chn_stock::daily::cne6::dividend_yield::create(),
        crate::barra::chn_stock::daily::cne6::growth::create(),
        crate::barra::chn_stock::daily::cne6::liquidity::create(),
        crate::barra::chn_stock::daily::cne6::momentum::create(),
        crate::barra::chn_stock::daily::cne6::quality::create(),
        crate::barra::chn_stock::daily::cne6::sentiment::create(),
        crate::barra::chn_stock::daily::cne6::size::create(),
        crate::barra::chn_stock::daily::cne6::value::create(),
        crate::barra::chn_stock::daily::cne6::volatility::create(),
    ]
}
