"""Public-fund catalog interfaces; daily queries bound memory independently of history length."""
from datetime import timedelta

import numpy as np

from .fund_downloader import (
    _FundDownloader, _concat_preserve_schema, atomic_write, normalize,
    parse_date, today, unique_query, writer_lock,
)


class _FundSnapshotDownloader(_FundDownloader):
    numeric = ()
    key = ("ts_code",)
    paginated = True

    def _normalize(self, frame):
        return normalize(frame, self.fields, self.numeric, required_text=self.key)

    def sync(self):
        with writer_lock(self.save_dir):
            if self.paginated:
                pages = list(self._pages())
            else:
                page = self._request()
                pages = [] if page.empty else [self._normalize(page)]
            if not pages:
                raise RuntimeError(f"{self.endpoint}: refusing to replace a snapshot with empty data")
            result = unique_query(_concat_preserve_schema(pages), list(self.key), self.fields)
            result = self._normalize(result)
            result = result.sort_values(list(self.key)).reset_index(drop=True)
            result["fetch_date"] = np.int32(today())
            atomic_write(result, self.save_dir / "snapshots" / f"{today()}.parquet")
            atomic_write(result, self.save_dir / f"{self.endpoint}.parquet")
            self.logger.info(f"{self.endpoint} saved {len(result)} snapshot rows")


class FundCompanyDownloader(_FundSnapshotDownloader):
    fields = ("name,shortname,short_enname,province,city,address,phone,office,website,"
              "chairman,manager,reg_capital,setup_date,end_date,employees,main_business,"
              "org_code,credit_code").split(",")
    numeric = ("reg_capital", "employees")
    key = ("name",)
    paginated = False  # Official contract: no inputs, one complete response.

    def __init__(self):
        super().__init__("fund_company", "fund_company_dir", "fund_data/company", 1000, 180)


class FundBenchmarkDownloader(_FundSnapshotDownloader):
    fields = "ts_code,symbol,name,fullname,bmk_level,bmk_type,bmk_src,idx_type".split(",")

    def __init__(self):
        super().__init__("mkt_idx_bmk", "fund_benchmark_dir", "fund_data/benchmark", 5000, 180)
        self.page_limit = min(self.page_limit, 5000)


class _FundDatedDownloader(_FundDownloader):
    numeric = ()
    key = ("ts_code", "trade_date")
    date_field = "trade_date"
    annual = False

    def _normalize(self, frame):
        return normalize(frame, self.fields, self.numeric,
                         required_dates=(self.date_field,), required_text=("ts_code",))

    def sync(self, mode="historical", start_year=2009, start_date=None, end_date=None, lookback_days=None):
        end = parse_date(end_date or today())
        if mode == "historical":
            begin = parse_date(start_date or f"{int(start_year):04d}0101")
        elif mode == "incremental":
            days = self.lookback_days if lookback_days is None else int(lookback_days)
            if days < 0:
                raise ValueError("Negative fund lookback")
            begin = parse_date(start_date or end.strftime("%Y%m%d")) - timedelta(days=days)
        else:
            raise ValueError(f"Unknown fund mode: {mode}")
        if begin > end:
            raise ValueError("Fund start date exceeds end date")
        with writer_lock(self.save_dir):
            # Announcements and NAV can occur on non-trading days as well.
            while begin <= end:
                date = begin.strftime("%Y%m%d")
                pages = list(self._pages(**{self.date_field: date}))
                if pages:
                    frame = unique_query(_concat_preserve_schema(pages), list(self.key), self.fields)
                    if not frame[self.date_field].eq(int(date)).all():
                        raise ValueError(f"{self.endpoint}: response outside requested {self.date_field}")
                    name = str(begin.year) if self.annual else date
                    self._merge_partition(frame, self.save_dir / f"{name}.parquet", list(self.key))
                begin += timedelta(days=1)


class FundManagerDownloader(_FundDatedDownloader):
    fields = "ts_code,ann_date,name,gender,birth_year,edu,nationality,begin_date,end_date,resume".split(",")
    key = ("ts_code", "ann_date", "name", "begin_date")
    date_field = "ann_date"
    annual = True

    def __init__(self):
        super().__init__("fund_manager", "fund_manager_dir", "fund_data/manager", 5000, 180)
        self.page_limit = min(self.page_limit, 5000)


class FundShareDownloader(_FundDatedDownloader):
    fields = "ts_code,trade_date,fd_share".split(",")
    numeric = ("fd_share",)

    def __init__(self):
        super().__init__("fund_share", "fund_share_dir", "fund_data/share", 2000, 180)
        self.page_limit = min(self.page_limit, 2000)


class FundNavDownloader(_FundDatedDownloader):
    fields = "ts_code,ann_date,nav_date,unit_nav,accum_nav,accum_div,net_asset,total_netasset,adj_nav".split(",")
    numeric = tuple(fields[3:])
    key = ("ts_code", "nav_date", "ann_date")
    date_field = "nav_date"

    def __init__(self):
        super().__init__("fund_nav", "fund_nav_dir", "fund_data/nav", 1000, 180)


class FundDividendDownloader(_FundDatedDownloader):
    fields = ("ts_code,ann_date,imp_anndate,base_date,div_proc,record_date,ex_date,pay_date,"
              "earpay_date,net_ex_date,div_cash,base_unit,ear_distr,ear_amount,account_date,base_year").split(",")
    numeric = ("div_cash", "base_unit", "ear_distr", "ear_amount")
    key = ("ts_code", "ann_date", "base_date", "div_proc", "ex_date")
    date_field = "ann_date"
    annual = True

    def __init__(self):
        super().__init__("fund_div", "fund_div_dir", "fund_data/dividend", 1000, 180)


class FundFactorProDownloader(_FundDatedDownloader):
    fields = ("ts_code,trade_date,trade_date_doris,open,high,low,close,pre_close,change,pct_change,vol,amount,"
              "asi_bfq,asit_bfq,atr_bfq,bbi_bfq,bias1_bfq,bias2_bfq,bias3_bfq,boll_lower_bfq,boll_mid_bfq,"
              "boll_upper_bfq,brar_ar_bfq,brar_br_bfq,cci_bfq,cr_bfq,dfma_dif_bfq,dfma_difma_bfq,"
              "dmi_adx_bfq,dmi_adxr_bfq,dmi_mdi_bfq,dmi_pdi_bfq,downdays,updays,dpo_bfq,madpo_bfq,"
              "ema_bfq_10,ema_bfq_20,ema_bfq_250,ema_bfq_30,ema_bfq_5,ema_bfq_60,ema_bfq_90,emv_bfq,"
              "maemv_bfq,expma_12_bfq,expma_50_bfq,kdj_bfq,kdj_d_bfq,kdj_k_bfq,ktn_down_bfq,ktn_mid_bfq,"
              "ktn_upper_bfq,lowdays,topdays,ma_bfq_10,ma_bfq_20,ma_bfq_250,ma_bfq_30,ma_bfq_5,ma_bfq_60,"
              "ma_bfq_90,macd_bfq,macd_dea_bfq,macd_dif_bfq,mass_bfq,ma_mass_bfq,mfi_bfq,mtm_bfq,mtmma_bfq,"
              "obv_bfq,psy_bfq,psyma_bfq,roc_bfq,maroc_bfq,rsi_bfq_12,rsi_bfq_24,rsi_bfq_6,taq_down_bfq,"
              "taq_mid_bfq,taq_up_bfq,trix_bfq,trma_bfq,vr_bfq,wr_bfq,wr1_bfq,xsii_td1_bfq,xsii_td2_bfq,"
              "xsii_td3_bfq,xsii_td4_bfq").split(",")
    numeric = tuple(fields[3:])

    def __init__(self):
        super().__init__("fund_factor_pro", "fund_factor_pro_dir", "fund_data/factor_pro", 8000, 30)
        self.page_limit = min(self.page_limit, 8000)
