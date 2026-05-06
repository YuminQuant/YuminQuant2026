pub const DP_POS_PRICE_CORR_RAW_ID: &str = "daily_dp_pos_price_corr";
pub const DP_NEG_PRICE_CORR_RAW_ID: &str = "daily_dp_neg_price_corr";
pub const DP_POS_NEXT_DP_POS_CORR_RAW_ID: &str = "daily_dp_pos_next_dp_pos_corr";
pub const DP_NEG_NEXT_DP_NEG_CORR_RAW_ID: &str = "daily_dp_neg_next_dp_neg_corr";

pub const OPEN_AUCTION_TURNOVER_RAW_ID: &str = "daily_open_auction_turnover";

pub const PM_CO_RAW_ID: &str = "daily_pm_co";
pub const PM_SMART_TURNOVER_RAW_ID: &str = "daily_pm_smart_turnover";
pub const LAST30_TURNOVER_RAW_ID: &str = "daily_last30m_turnover";

pub const TURNOVER_VOLATILITY_RAW_ID: &str = "daily_turnover_rate_volatility";
pub const INTRADAY_RETURN_VOLATILITY_RAW_ID: &str = "daily_intraday_return_volatility";
pub const VOLUME_CV_RAW_ID: &str = "daily_volume_cv";

pub const SUBRS_5MIN_RAW_ID: &str = "daily_subrs_5min";
pub const SUBRK_5MIN_RAW_ID: &str = "daily_subrk_5min";
pub const SUBRHS_5MIN_RAW_ID: &str = "daily_subrhs_5min";
pub const SUBRHT_5MIN_RAW_ID: &str = "daily_subrht_5min";

pub const RV_5MIN_RAW_ID: &str = "daily_rv_5min";
pub const VAR90_5MIN_RAW_ID: &str = "daily_var90_5min";
pub const VAR95_5MIN_RAW_ID: &str = "daily_var95_5min";
pub const CVAR90_5MIN_RAW_ID: &str = "daily_cvar90_5min";
pub const CVAR95_5MIN_RAW_ID: &str = "daily_cvar95_5min";
pub const VAR90_RT_5MIN_RAW_ID: &str = "daily_var90_rt_5min";
pub const VAR95_RT_5MIN_RAW_ID: &str = "daily_var95_rt_5min";
pub const CVAR90_RT_5MIN_RAW_ID: &str = "daily_cvar90_rt_5min";
pub const CVAR95_RT_5MIN_RAW_ID: &str = "daily_cvar95_rt_5min";
pub const ID_RV_5MIN_RAW_ID: &str = "daily_id_rv_5min";
pub const ID_VAR90_5MIN_RAW_ID: &str = "daily_id_var90_5min";
pub const ID_VAR95_5MIN_RAW_ID: &str = "daily_id_var95_5min";
pub const ID_CVAR90_5MIN_RAW_ID: &str = "daily_id_cvar90_5min";
pub const ID_CVAR95_5MIN_RAW_ID: &str = "daily_id_cvar95_5min";
pub const ID_VAR90_RT_5MIN_RAW_ID: &str = "daily_id_var90_rt_5min";
pub const ID_VAR95_RT_5MIN_RAW_ID: &str = "daily_id_var95_rt_5min";
pub const ID_CVAR90_RT_5MIN_RAW_ID: &str = "daily_id_cvar90_rt_5min";
pub const ID_CVAR95_RT_5MIN_RAW_ID: &str = "daily_id_cvar95_rt_5min";

pub const RTN5_MEAN_RAW_ID: &str = "daily_rtn5_mean";
pub const REAL_VAR_RAW_ID: &str = "daily_real_var";
pub const RTN_SKEW_RAW_ID: &str = "daily_rtn_skew";
pub const RTN_KURT_RAW_ID: &str = "daily_rtn_kurt";
pub const RV_UP_RAW_ID: &str = "daily_rv_up";
pub const RV_DOWN_RAW_ID: &str = "daily_rv_down";
pub const RV_UMD_RAW_ID: &str = "daily_rv_umd";
pub const NOS_SW_RAW_ID: &str = "daily_nos_sw";
pub const NOS_GS_RAW_ID: &str = "daily_nos_gs";
pub const CPR_SW_RAW_ID: &str = "daily_cpr_sw";

pub const EX_RTN_MAX_VAL_RAW_ID: &str = "daily_ex_rtn_max_val";
pub const EX_RTN_MAX_FRE_RAW_ID: &str = "daily_ex_rtn_max_fre";
pub const EX_RTN_MIN_VAL_RAW_ID: &str = "daily_ex_rtn_min_val";
pub const EX_RTN_MIN_FRE_RAW_ID: &str = "daily_ex_rtn_min_fre";
pub const GMM_MEAN_RAW_ID: &str = "daily_gmm_mean";
pub const GMM_MEAN2WGT_RAW_ID: &str = "daily_gmm_mean2wgt";
pub const GMM_MEANDIF_RAW_ID: &str = "daily_gmm_meandif";
pub const GMM_MEANDIF2WGTDIF_RAW_ID: &str = "daily_gmm_meandif2wgtdif";

pub const LOGVOL_SKEW_RAW_ID: &str = "daily_logvol_skew";
pub const LOGVOL_90TAIL_RAW_ID: &str = "daily_logvol_90tail";
pub const LOGVOL_10TAIL_RAW_ID: &str = "daily_logvol_10tail";
pub const VOLROC_SKEW_RAW_ID: &str = "daily_volroc_skew";
pub const VOLROC_KURT_RAW_ID: &str = "daily_volroc_kurt";
pub const CUMSUMVOL_MEAN_RAW_ID: &str = "daily_cumsumvol_mean";
pub const CUMSUMVOL_STD_RAW_ID: &str = "daily_cumsumvol_std";
pub const VOL_ENTROPY_SHAPE_RAW_ID: &str = "daily_vol_entropy_shape";
pub const VOL_MAXMEAN_RAW_ID: &str = "daily_vol_maxmean";
pub const VOL_MAXSTD_RAW_ID: &str = "daily_vol_maxstd";
pub const VSA_RATIO_RAW_ID: &str = "daily_vsa_ratio";
pub const VSA_LOW2MAX_RAW_ID: &str = "daily_vsa_low2max";
pub const VSA_HIGH2MIN_RAW_ID: &str = "daily_vsa_high2min";

pub const RTN_FOC_RAW_ID: &str = "daily_rtn_foc";
pub const VOL_FOC_RAW_ID: &str = "daily_vol_foc";
pub const RTN_DW_RAW_ID: &str = "daily_rtn_dw";
pub const VOL_DW_RAW_ID: &str = "daily_vol_dw";
pub const RTN_RHO_RAW_ID: &str = "daily_rtn_rho";
pub const VOL_RHO_RAW_ID: &str = "daily_vol_rho";
pub const RTN_LBQ_RAW_ID: &str = "daily_rtn_lbq";
pub const VOL_LBQ_RAW_ID: &str = "daily_vol_lbq";
pub const HIGH_STD_RTN_MEAN_RAW_ID: &str = "daily_high_std_rtn_mean";
pub const RTN_COND_VAR_RAW_ID: &str = "daily_rtn_cond_var";
pub const FLASH_CRASH_PROB_RAW_ID: &str = "daily_flash_crash_prob";

pub const RVC_COR_RAW_ID: &str = "daily_rvc_cor";
pub const RHL_COR_RAW_ID: &str = "daily_rhl_cor";
pub const VOH_COR_RAW_ID: &str = "daily_voh_cor";
pub const VOL_COR_RAW_ID: &str = "daily_vol_cor";
pub const TE_V2R_RAW_ID: &str = "daily_te_v2r";
pub const TE_R2V_RAW_ID: &str = "daily_te_r2v";
pub const CUTVOL_RTN_MEAN_RAW_ID: &str = "daily_cutvol_rtn_mean";
pub const CUTVOL_RTN_VAR_RAW_ID: &str = "daily_cutvol_rtn_var";
pub const CUTVOL_TIME_MEAN_RAW_ID: &str = "daily_cutvol_time_mean";
pub const CUTVOL_TIME_VAR_RAW_ID: &str = "daily_cutvol_time_var";
pub const CUTVOL_TIME_COR_RAW_ID: &str = "daily_cutvol_time_cor";
pub const CUTVOL_ENTROPY_RAW_ID: &str = "daily_cutvol_entropy";
