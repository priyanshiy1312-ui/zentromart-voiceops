"""Read-only helpers for reference data (pincodes, holidays, products) and the date rules.

NOTE: customers.csv / orders.csv are deliberately NOT exposed here. Order details must go
through the provided `lookup_order` tool, which enforces caller verification.
"""
import csv
from datetime import date, datetime, timedelta
from pathlib import Path

from . import config

_DATE_FORMATS = ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d", "%d %b %Y", "%d %B %Y")


def parse_date(value):
    if isinstance(value, date):
        return value
    value = str(value).strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    try:
        return date.fromisoformat(value[:10])
    except ValueError as exc:
        raise ValueError(f"Unrecognised date: {value!r}") from exc


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))  # everything stays a string (IDs/pincodes!)


class DataStore:
    def __init__(self, data_dir=None):
        self.data_dir = Path(data_dir or config.DATA_DIR)
        self.pincodes = {r.get("pincode", "").strip(): r for r in read_csv(self.data_dir / "pincodes.csv")}
        self.products = read_csv(self.data_dir / "products.csv")
        self.holidays = set()
        for row in read_csv(self.data_dir / "holidays.csv"):
            try:
                self.holidays.add(parse_date(row.get("date", "")))
            except ValueError:
                pass

    # ---- reference lookups -------------------------------------------------
    def check_pincode(self, pincode):
        row = self.pincodes.get(str(pincode).strip())
        return dict(row) if row else {"error": f"pincode {pincode} not found in pincodes table"}

    def product_info(self, query, limit=3):
        q = str(query).strip().lower()
        hits = []
        for row in self.products:
            blob = " ".join(str(v).lower() for v in row.values())
            if q and (q == str(row.get("sku", "")).lower() or q in blob):
                hits.append(row)
        return hits[:limit] or [{"error": f"no product matched {query!r}"}]

    # ---- date rules (business days = Mon-Sat excluding holidays.csv) -------
    def is_business_day(self, d):
        return d.weekday() != 6 and d not in self.holidays

    def add_business_days(self, start, n):
        d, left = parse_date(start), int(n)
        while left > 0:
            d += timedelta(days=1)
            if self.is_business_day(d):
                left -= 1
        return d

    def business_days_between(self, start, end):
        """Business days elapsed AFTER `start` up to and including `end`."""
        a, b = parse_date(start), parse_date(end)
        sign, count = (1, 0) if b >= a else (-1, 0)
        lo, hi = (a, b) if sign == 1 else (b, a)
        d = lo
        while d < hi:
            d += timedelta(days=1)
            if self.is_business_day(d):
                count += 1
        return sign * count

    def date_calc(self, op, start=None, days=None, end=None):
        """Deterministic date maths so the LLM never does calendar arithmetic itself."""
        today = config.TODAY
        start = parse_date(start) if start else today
        if op == "add_business_days":
            return {"result_date": self.add_business_days(start, days).isoformat()}
        if op == "add_calendar_days":
            return {"result_date": (start + timedelta(days=int(days))).isoformat()}
        if op == "business_days_between":
            return {"business_days": self.business_days_between(start, parse_date(end) if end else today)}
        if op == "calendar_days_between":
            return {"calendar_days": ((parse_date(end) if end else today) - start).days}
        return {"error": f"unknown op {op!r}"}
