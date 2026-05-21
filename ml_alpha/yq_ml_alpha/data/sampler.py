from __future__ import annotations

from yq_ml_alpha.calendar import TradingCalendar


def sample_dates(calendar: TradingCalendar, date_range: tuple[int, int], frequency: str) -> list[int]:
    dates = calendar.between(date_range[0], date_range[1])
    frequency = frequency.lower().strip()
    if frequency == "daily":
        return dates
    if frequency in {"weekly", "weekly_end"}:
        return _period_ends(dates, lambda date: date // 10000 * 100 + _week_number(date))
    if frequency in {"monthly", "monthly_end"}:
        return _period_ends(dates, lambda date: date // 100)
    if frequency in {"semiannual", "semiannual_end", "halfyear", "halfyear_end"}:
        return _period_ends(dates, _semiannual_key)
    if frequency in {"annual", "annual_end", "yearly", "yearly_end", "year_end"}:
        return _period_ends(dates, lambda date: date // 10000)
    step = _fixed_step(frequency)
    if step is not None:
        return dates[::step]
    raise ValueError(f"unsupported sample frequency: {frequency}")


def refit_dates(calendar: TradingCalendar, predict_dates: list[int], frequency: str) -> list[int]:
    if not predict_dates:
        return []
    frequency = frequency.lower().strip()
    if frequency == "daily":
        return predict_dates
    if frequency in {"monthly", "monthly_end"}:
        return sample_dates(calendar, (predict_dates[0], predict_dates[-1]), "monthly_end")
    if frequency in {"weekly", "weekly_end"}:
        return sample_dates(calendar, (predict_dates[0], predict_dates[-1]), "weekly_end")
    if frequency in {"semiannual", "semiannual_end", "halfyear", "halfyear_end"}:
        return sample_dates(calendar, (predict_dates[0], predict_dates[-1]), "semiannual_end")
    if frequency in {"annual", "annual_end", "yearly", "yearly_end", "year_end"}:
        return sample_dates(calendar, (predict_dates[0], predict_dates[-1]), "annual_end")
    if _fixed_step(frequency) is not None:
        return sample_dates(calendar, (predict_dates[0], predict_dates[-1]), frequency)
    return [predict_dates[0]]


def _fixed_step(frequency: str) -> int | None:
    if frequency.isdigit():
        step = int(frequency)
    elif frequency.startswith("every_") and frequency.endswith("_days"):
        step = int(frequency[len("every_") : -len("_days")])
    else:
        return None
    if step <= 0:
        raise ValueError("fixed-day sample frequency requires N > 0")
    return step


def _period_ends(dates: list[int], key_fn) -> list[int]:
    output = []
    for idx, date in enumerate(dates):
        next_date = dates[idx + 1] if idx + 1 < len(dates) else None
        if next_date is None or key_fn(next_date) != key_fn(date):
            output.append(date)
    return output


def _week_number(yyyymmdd: int) -> int:
    import datetime as dt

    text = str(yyyymmdd)
    date = dt.date(int(text[:4]), int(text[4:6]), int(text[6:]))
    year, week, _ = date.isocalendar()
    return year * 100 + week


def _semiannual_key(yyyymmdd: int) -> int:
    year = yyyymmdd // 10000
    month = (yyyymmdd // 100) % 100
    half = 1 if month <= 6 else 2
    return year * 10 + half
