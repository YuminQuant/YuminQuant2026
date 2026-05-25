use crate::factor::common::kyzq_peak_valley::PeakValleyMetric;

crate::define_kyzq_peak_valley_factor!(
    StockDailyVolumeValleyRelativeVwap,
    PeakValleyMetric::VolumeValleyRelativeVwap
);
