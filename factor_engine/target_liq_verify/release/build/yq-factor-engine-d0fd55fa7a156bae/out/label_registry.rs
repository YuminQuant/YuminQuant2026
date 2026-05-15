use std::collections::HashMap;

use crate::label::Label;

pub fn all_labels() -> Vec<Box<dyn Label>> {
    vec![
        crate::label::chn_stock::daily::future_close_return_1d::create(),
        crate::label::chn_stock::daily::future_close_return_20d::create(),
        crate::label::chn_stock::daily::future_close_return_5d::create(),
        crate::label::chn_stock::daily::future_open_10m_vwap_return_1d::create(),
        crate::label::chn_stock::daily::future_open_10m_vwap_return_20d::create(),
        crate::label::chn_stock::daily::future_open_10m_vwap_return_5d::create(),
        crate::label::chn_stock::daily::future_open_5m_vwap_return_1d::create(),
        crate::label::chn_stock::daily::future_open_5m_vwap_return_20d::create(),
        crate::label::chn_stock::daily::future_open_5m_vwap_return_5d::create(),
        crate::label::chn_stock::daily::future_open_return_1d::create(),
        crate::label::chn_stock::daily::future_open_return_20d::create(),
        crate::label::chn_stock::daily::future_open_return_5d::create(),
        crate::label::chn_stock::daily::future_vwap_return_1d::create(),
        crate::label::chn_stock::daily::future_vwap_return_20d::create(),
        crate::label::chn_stock::daily::future_vwap_return_5d::create(),
    ]
}

pub fn label_map() -> HashMap<String, Box<dyn Label>> {
    let mut items = HashMap::new();
    for item in all_labels() {
        let key = item.spec().registry_key();
        items.insert(key, item);
    }
    items
}
