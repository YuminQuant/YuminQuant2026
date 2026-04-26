import os

import pandas as pd

from data_manager.core import BaseDownloader, ConfigManager


IS_NEW_STATUSES = ("N", "Y")


class _PagedIndexMemberDownloader(BaseDownloader):
    dataset_name = "index_member"
    output_filename = "members.parquet"
    sort_candidates = (
        "ts_code",
        "l1_code",
        "l2_code",
        "l3_code",
        "in_date",
        "out_date",
        "is_new",
    )

    def _fetch_page(self, is_new, offset):
        raise NotImplementedError

    def _fetch_all_pages(self, is_new):
        frames = []
        offset = 0

        while True:
            try:
                df = self._fetch_page(is_new=is_new, offset=offset)
            except Exception as exc:
                self.logger.error(
                    f"{self.dataset_name} fetch failed for is_new={is_new}, "
                    f"offset={offset}: {exc}"
                )
                break

            if df is None or df.empty:
                break

            frames.append(df)
            if len(df) < self.page_limit:
                break

            offset += self.page_limit
            self.safe_sleep()
            self.logger.info(
                f"{self.dataset_name} fetched is_new={is_new}, next offset={offset}"
            )

        return frames

    def _combine_frames(self, frames):
        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        combined.drop_duplicates(inplace=True)
        sort_columns = [col for col in self.sort_candidates if col in combined.columns]
        if sort_columns:
            combined.sort_values(by=sort_columns, inplace=True)
        combined.reset_index(drop=True, inplace=True)
        return combined

    def sync(self):
        self.logger.info(f"=== Sync {self.dataset_name} members ===")
        frames = []
        for is_new in IS_NEW_STATUSES:
            frames.extend(self._fetch_all_pages(is_new))

        combined = self._combine_frames(frames)
        if combined.empty:
            self.logger.warning(f"No {self.dataset_name} member data fetched.")
            return

        file_path = os.path.join(self.save_dir, self.output_filename)
        combined.to_parquet(file_path, index=False)
        self.logger.info(
            f"Saved {len(combined)} {self.dataset_name} member rows to {file_path}"
        )


class SWMemberDownloader(_PagedIndexMemberDownloader):
    """Download and overwrite Shenwan index member history."""

    dataset_name = "sw_member"
    output_filename = "sw_members.parquet"

    def __init__(self):
        config = ConfigManager().config
        super().__init__(rate_limit=config["api"]["rate_limits"].get("index_member_sw", 500))
        self.page_limit = config["api"]["page_limits"].get("index_member_sw", 2000)
        self.save_dir = self.get_full_path_and_ensure_dir("index_member_sw_dir")

    def _fetch_page(self, is_new, offset):
        return self.pro.index_member_all(
            is_new=is_new,
            limit=self.page_limit,
            offset=offset,
        )


class CIMemberDownloader(_PagedIndexMemberDownloader):
    """Download and overwrite CITIC index member history."""

    dataset_name = "ci_member"
    output_filename = "ci_members.parquet"

    def __init__(self):
        config = ConfigManager().config
        super().__init__(rate_limit=config["api"]["rate_limits"].get("index_member_ci", 500))
        self.page_limit = config["api"]["page_limits"].get("index_member_ci", 4000)
        self.save_dir = self.get_full_path_and_ensure_dir("index_member_ci_dir")

    def _fetch_page(self, is_new, offset):
        return self.pro.ci_index_member(
            is_new=is_new,
            limit=self.page_limit,
            offset=offset,
        )
