#!/usr/bin/env python3
"""Scrape monthly Taiwan Labor Insurance Fund, population, and pensioner data.

The script downloads the official monthly ``勞工保險基金`` report, converts it
to text with Poppler's ``pdftotext``, and extracts the exact month-end total from
the fund's investment/allocation table. It then joins (1) month-end registered
population from the Ministry of the Interior and (2) the number of Labor
Insurance old-age-annuity recipients from Bureau of Labor Insurance reports.

The Labor Insurance Fund included occupational-accident insurance through
2022-04. From 2022-05 onward, official reports state that it contains ordinary-
accident insurance only because occupational-accident insurance was separated
into its own fund under the Occupational Accident Insurance and Protection Act.
"""

from __future__ import annotations

import argparse
import csv
import http.cookiejar
import io
import re
import subprocess
import time
import unicodedata
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from lxml import html


BASE = "https://www.blf.gov.tw"
LIST_URL = BASE + "/49200/49255/49281/49285/49289/lpsimplelist"
USER_AGENT = "Mozilla/5.0 (compatible; academic-data-collection/1.0)"
BLI_MONTHLY_URL = "https://www.bli.gov.tw/ReportMonth/{roc_year}"
BLI_REPORT_BASE = "https://events.bli.gov.tw/report/"
MOI_DOWNLOAD_PAGE = (
    "https://www.ris.gov.tw/info-popudata/app/awFastDownload/toMain_panel"
)
MOI_DOWNLOAD_URL = "https://www.ris.gov.tw/info-popudata/app/awFastDownload/view"


@dataclass(frozen=True)
class Report:
    year: int
    month: int
    roc_year: int
    url: str
    title: str

    @property
    def period(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"


def get_bytes(url: str, attempts: int = 4) -> bytes:
    parts = urllib.parse.urlsplit(url)
    safe_url = urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, urllib.parse.quote(parts.path), parts.query, parts.fragment)
    )
    request = urllib.request.Request(safe_url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return response.read()
        except Exception as error:  # retry transient server/network failures
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(1.5 * (attempt + 1))
    assert last_error is not None
    raise last_error


def resolve_report_pdf(item_url: str, period: str) -> str:
    """Resolve a monthly archive item to its official main fund PDF.

    Recent archive items link directly to a consolidated PDF. Older items link
    to a detail page with several attachments; in those pages the relevant
    attachment is explicitly titled ``勞工保險基金投資運用``.
    """
    if ".pdf" in urllib.parse.urlsplit(item_url).path.lower():
        return item_url

    page = html.fromstring(get_bytes(item_url))
    candidates: list[str] = []
    for anchor in page.xpath('//a[contains(translate(@href,"PDF","pdf"), ".pdf")]'):
        label = normalize_label("".join(anchor.itertext()))
        is_main_report = bool(
            re.search(r"勞(?:工保險|保)基金投資運用", label)
            or re.search(r"勞保基金規模.*投資績效.*資產配置", label)
        )
        if not is_main_report or "股票" in label:
            continue
        url = urllib.parse.urljoin(BASE, anchor.get("href"))
        # The legacy pages sometimes list downloadable and inline variants.
        canonical = url.replace("?mediaDL=true", "")
        if canonical not in candidates:
            candidates.append(canonical)
    if not candidates:
        raise RuntimeError(f"{period}: Labor Insurance Fund PDF was not found")
    return candidates[0] + "?mediaDL=true"


def discover_reports(start_year: int, start_month: int) -> list[Report]:
    landing = html.fromstring(get_bytes(LIST_URL))
    year_to_attribute: dict[int, str] = {}
    for anchor in landing.xpath('//a[contains(@href, "q_attribute=")]'):
        label = "".join(anchor.itertext()).strip()
        match = re.fullmatch(r"(\d{2,3})年", label)
        if not match:
            continue
        roc_year = int(match.group(1))
        query = urllib.parse.urlparse(anchor.get("href")).query
        attribute = urllib.parse.parse_qs(query).get("q_attribute", [None])[0]
        if attribute:
            year_to_attribute[roc_year] = attribute

    reports: dict[tuple[int, int], Report] = {}

    def fetch_year(item: tuple[int, str]) -> tuple[int, bytes]:
        roc_year, attribute = item
        query = urllib.parse.urlencode(
            {"Page": 1, "PageSize": 100, "q_attribute": attribute}
        )
        return roc_year, get_bytes(f"{LIST_URL}?{query}")

    selected = [
        (roc_year, attribute)
        for roc_year, attribute in sorted(year_to_attribute.items())
        if roc_year + 1911 >= start_year
    ]
    pages: list[tuple[int, bytes]] = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        pages = list(pool.map(fetch_year, selected))

    for roc_year, page_bytes in pages:
        year = roc_year + 1911
        page = html.fromstring(page_bytes)
        for anchor in page.xpath('//div[contains(@class, "item_title")]/a'):
            href = anchor.get("href", "")
            title = "".join(anchor.itertext()).strip()
            month_match = re.search(r"(\d{1,2})月", title)
            if not month_match:
                continue
            month = int(month_match.group(1))
            if not 1 <= month <= 12:
                continue
            if (year, month) < (start_year, start_month):
                continue
            period = f"{year:04d}-{month:02d}"
            item_url = urllib.parse.urljoin(BASE, href)
            reports[(year, month)] = Report(
                year=year,
                month=month,
                roc_year=roc_year,
                url=item_url,
                title=title,
            )

    ordered = [reports[key] for key in sorted(reports)]

    def resolve(report: Report) -> Report:
        return Report(
            year=report.year,
            month=report.month,
            roc_year=report.roc_year,
            url=resolve_report_pdf(report.url, report.period),
            title=report.title,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(resolve, ordered))


def download_report(report: Report, pdf_dir: Path) -> Path:
    destination = pdf_dir / f"{report.period}.pdf"
    if not destination.exists() or destination.stat().st_size == 0:
        destination.write_bytes(get_bytes(report.url))
    return destination


def pdf_to_text(pdf_path: Path, text_dir: Path) -> str:
    text_path = text_dir / f"{pdf_path.stem}.txt"
    if not text_path.exists() or text_path.stat().st_mtime < pdf_path.stat().st_mtime:
        subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), str(text_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    return unicodedata.normalize("NFKC", text_path.read_text(errors="replace"))


TOTAL_PATTERN = re.compile(
    r"(?:合\s*計|總\s*計)\s+([0-9][0-9, .]{6,30}?)\s*(?:元)?\s+100(?:\.0+)?\s*%?"
)
SCALE_PATTERN = re.compile(
    r"基金\s*運用\s*規模[^\n]{0,120}?"
    r"(?:(\d+)\s*兆\s*)?([0-9,]+)\s*億(?:\s*([0-9,]+)\s*萬)?(?:元)?",
)


def extract_exact_balance(text: str) -> int | None:
    """Extract the first 100%-total in the Labor Insurance Fund report."""
    for match in TOTAL_PATTERN.finditer(text):
        digits = re.sub(r"[^0-9]", "", match.group(1))
        if digits:
            value = int(digits)
            if 10_000_000_000 <= value <= 10_000_000_000_000:
                return value
    return None


def extract_reported_scale(text: str) -> int | None:
    match = SCALE_PATTERN.search(text)
    if not match:
        return None
    trillion = int(match.group(1) or 0)
    hundred_million = int(match.group(2).replace(",", ""))
    ten_thousand = int((match.group(3) or "0").replace(",", ""))
    return (
        trillion * 1_000_000_000_000
        + hundred_million * 100_000_000
        + ten_thousand * 10_000
    )


def normalize_label(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value)))


def month_range(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    year, month = start
    result: list[tuple[int, int]] = []
    while (year, month) <= end:
        result.append((year, month))
        if month == 12:
            year, month = year + 1, 1
        else:
            month += 1
    return result


def discover_old_age_pension_urls(
    periods: list[tuple[int, int]],
) -> dict[str, str | None]:
    """Find official monthly CSVs for Labor Insurance old-age annuities."""
    wanted = {f"{year:04d}-{month:02d}" for year, month in periods}
    urls: dict[str, str | None] = {}

    # The annuity took effect on 2009-01-01. No January payment report exists,
    # and there could be no recipients before the program opened.
    for period in wanted:
        if period <= "2009-01":
            urls[period] = None

    # The 2009-2010 archive predates the current year-index pages. Its regional
    # table has a stable file code and contains the exact old-age total.
    for year, month in periods:
        if (year, month) < (2009, 2) or year > 2010:
            continue
        roc_month = f"{year - 1911:03d}{month:02d}"
        urls[f"{year:04d}-{month:02d}"] = urllib.parse.urljoin(
            BLI_REPORT_BASE,
            f"attachment_file/report/month/{roc_month}/a12010.csv",
        )

    roc_years = sorted({year - 1911 for year, _ in periods if year >= 2011})
    for roc_year in roc_years:
        page = html.fromstring(get_bytes(BLI_MONTHLY_URL.format(roc_year=roc_year)))
        for anchor in page.xpath('//a[contains(@href, "reportM.aspx")]'):
            title = normalize_label("".join(anchor.itertext()))
            if not (
                "勞工保險" in title
                and "年金給付" in title
                and "地區" in title
                and "職業災害" not in title
                and "普通事故" not in title
            ):
                continue
            parsed = urllib.parse.urlparse(anchor.get("href"))
            query = urllib.parse.parse_qs(parsed.query)
            roc_month = query.get("m", [""])[0]
            file_code = query.get("f", [""])[0]
            match = re.fullmatch(r"(\d{3})(\d{2})", roc_month)
            if not match or not file_code:
                continue
            period = f"{int(match.group(1)) + 1911:04d}-{int(match.group(2)):02d}"
            if period in wanted:
                urls[period] = urllib.parse.urljoin(
                    BLI_REPORT_BASE,
                    f"attachment_file/report/month/{roc_month}/{file_code}.csv",
                )

    missing = sorted(wanted - set(urls))
    if missing:
        raise RuntimeError("Missing BLI monthly report URLs for: " + ", ".join(missing))
    return urls


def decode_bli_csv(data: bytes) -> list[list[str]]:
    for encoding in ("utf-8-sig", "big5", "cp950"):
        try:
            return list(csv.reader(io.StringIO(data.decode(encoding))))
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("BLI CSV", data, 0, 1, "unknown text encoding")


def extract_old_age_pension_recipients(data: bytes) -> int:
    """Extract the total recipient count under the old-age-annuity heading."""
    rows = decode_bli_csv(data)
    header_row = -1
    old_age_column = -1
    for row_number, row in enumerate(rows):
        for column, cell in enumerate(row):
            if normalize_label(cell) == "老年年金":
                header_row, old_age_column = row_number, column
                break
        if header_row >= 0:
            break
    if header_row < 0:
        raise ValueError("Old-age-annuity heading not found in BLI CSV")

    count_columns: list[int] = []
    for row in rows[header_row + 1:header_row + 5]:
        for column, cell in enumerate(row):
            if normalize_label(cell) in {"人數", "核付人數"}:
                count_columns.append(column)
    if count_columns:
        old_age_column = min(count_columns, key=lambda column: abs(column - old_age_column))

    for row in rows[header_row + 1:]:
        if old_age_column >= len(row):
            continue
        if "總計" not in normalize_label("".join(row[:old_age_column + 1])):
            continue
        digits = re.sub(r"[^0-9]", "", row[old_age_column])
        if digits:
            return int(digits)
    raise ValueError("Old-age-annuity total not found in BLI CSV")


def scrape_old_age_pension_recipients(
    periods: list[tuple[int, int]], cache: Path, workers: int,
) -> tuple[dict[str, int], dict[str, str]]:
    urls = discover_old_age_pension_urls(periods)
    cache.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    source_urls: dict[str, str] = {}

    def fetch(period: str, url: str) -> tuple[str, bytes]:
        path = cache / f"{period}.csv"
        if not path.exists() or path.stat().st_size == 0:
            path.write_bytes(get_bytes(url))
        return period, path.read_bytes()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fetch, period, url): (period, url)
            for period, url in urls.items() if url is not None
        }
        for future in as_completed(futures):
            period, url = futures[future]
            _, data = future.result()
            counts[period] = extract_old_age_pension_recipients(data)
            source_urls[period] = url

    for period, url in urls.items():
        if url is None:
            counts[period] = 0
            source_urls[period] = "https://www.bli.gov.tw/0105220.html"

    expected = {"2009-12": 65_632, "2010-12": 118_502}
    for period, value in expected.items():
        if period in counts and counts[period] != value:
            raise RuntimeError(
                f"{period}: BLI recipient total {counts[period]} != {value}"
            )
    return counts, source_urls


def moi_opener_and_token() -> tuple[urllib.request.OpenerDirector, str]:
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
    )
    request = urllib.request.Request(MOI_DOWNLOAD_PAGE, headers={"User-Agent": USER_AGENT})
    with opener.open(request, timeout=90) as response:
        landing = response.read().decode("utf-8", errors="replace")
    match = re.search(r'name="_csrf"\s+content="([^"]+)"', landing)
    if not match:
        raise RuntimeError("MOI download CSRF token was not found")
    return opener, match.group(1)


def download_moi_population_workbook(
    year: int, month: int, cache: Path,
    opener: urllib.request.OpenerDirector, token: str,
) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / f"{year:04d}-{month:02d}.xls"
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    roc_month = f"{year - 1911:03d}{month:02d}"
    payload = urllib.parse.urlencode(
        {"type": "xls", "m4c": "s000", "d5c": roc_month, "_csrf": token}
    ).encode()
    request = urllib.request.Request(
        MOI_DOWNLOAD_URL,
        data=payload,
        headers={"User-Agent": USER_AGENT, "Referer": MOI_DOWNLOAD_PAGE},
    )
    with opener.open(request, timeout=120) as response:
        data = response.read()
    if data[:8] != b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        raise RuntimeError(f"MOI returned a non-XLS response for {year:04d}-{month:02d}")
    destination.write_bytes(data)
    return destination


def extract_population_year(workbook: Path, year: int, last_month: int) -> dict[str, int]:
    excel = pd.ExcelFile(workbook)
    sheet = next(
        (name for name in excel.sheet_names if "年月別" in normalize_label(name)),
        None,
    )
    if sheet is None:
        raise RuntimeError(f"Year/month population sheet not found in {workbook}")
    frame = pd.read_excel(workbook, sheet_name=sheet, header=None)

    chinese_months = {
        "一月": 1, "二月": 2, "三月": 3, "四月": 4, "五月": 5, "六月": 6,
        "七月": 7, "八月": 8, "九月": 9, "十月": 10, "十一月": 11, "十二月": 12,
    }

    def parse_month(value: object) -> int | None:
        label = normalize_label(value)
        match = re.fullmatch(r"(\d{1,2})月", label)
        if match:
            return int(match.group(1))
        return chinese_months.get(label)

    # Report formats changed over time (Chinese versus Arabic month labels and
    # shifted columns). The current-year monthly block is the last sequential
    # 1,...,last_month run. The population itself is uniquely identifiable as
    # the row's 20-30 million integer.
    sequences: list[dict[int, int]] = []
    current: dict[int, int] = {}
    for _, row in frame.iterrows():
        month = parse_month(row.iloc[0])
        if month is None:
            continue
        if month == 1:
            current = {}
            sequences.append(current)
        if not current and month != 1:
            continue
        if month != len(current) + 1:
            current = {}
            continue
        candidates = []
        for value in row:
            if pd.isna(value):
                continue
            try:
                number = int(float(value))
            except (TypeError, ValueError):
                continue
            if 20_000_000 <= number <= 30_000_000:
                candidates.append(number)
        if len(candidates) != 1:
            raise RuntimeError(
                f"Could not identify unique population for {year:04d}-{month:02d}"
            )
        current[month] = candidates[0]

    complete = [sequence for sequence in sequences if set(range(1, last_month + 1)) <= set(sequence)]
    if not complete:
        raise RuntimeError(f"Current-year population block not found for {year}")
    selected = complete[-1]
    result = {
        f"{year:04d}-{month:02d}": selected[month]
        for month in range(1, last_month + 1)
    }
    expected = {f"{year:04d}-{month:02d}" for month in range(1, last_month + 1)}
    missing = sorted(expected - set(result))
    if missing:
        raise RuntimeError("Missing MOI population values for: " + ", ".join(missing))
    return result


def scrape_population(
    periods: list[tuple[int, int]], cache: Path,
) -> dict[str, int]:
    last_month_by_year: dict[int, int] = {}
    for year, month in periods:
        last_month_by_year[year] = max(last_month_by_year.get(year, 0), month)

    opener, token = moi_opener_and_token()
    populations: dict[str, int] = {}
    for year in sorted(last_month_by_year):
        last_month = last_month_by_year[year]
        workbook = download_moi_population_workbook(
            year, last_month, cache, opener, token
        )
        populations.update(extract_population_year(workbook, year, last_month))
    return populations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2008-12", help="first YYYY-MM")
    parser.add_argument("--output", default="labor_insurance_fund_monthly.csv")
    parser.add_argument("--cache", default="labor_insurance_cache")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    start_year, start_month = map(int, args.start.split("-"))
    cache = Path(args.cache)
    pdf_dir = cache / "pdf"
    text_dir = cache / "text"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    reports = discover_reports(start_year, start_month)
    if not reports:
        raise RuntimeError("No reports were discovered")

    periods = month_range(
        (start_year, start_month), (reports[-1].year, reports[-1].month)
    )
    populations = scrape_population(periods, cache / "population")
    pensioners, pension_source_urls = scrape_old_age_pension_recipients(
        periods, cache / "old_age_pension", args.workers
    )

    paths: dict[str, Path] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_report, report, pdf_dir): report for report in reports}
        for future in as_completed(futures):
            report = futures[future]
            paths[report.period] = future.result()

    rows: list[dict[str, object]] = []
    for report in reports:
        text = pdf_to_text(paths[report.period], text_dir)
        exact_balance = extract_exact_balance(text)
        reported_scale = extract_reported_scale(text)
        scope = (
            "普通事故及職業災害保險"
            if report.period <= "2022-04"
            else "普通事故保險（職災保險已分立）"
        )
        rows.append(
            {
                "date": report.period,
                "fund": "勞工保險基金",
                "fund_scope": scope,
                "fund_level_ntd": exact_balance,
                "fund_level_100m_ntd": (
                    round(exact_balance / 100_000_000, 4) if exact_balance else None
                ),
                "reported_fund_level_ntd": reported_scale,
                "taiwan_registered_population": populations[report.period],
                "labor_insurance_old_age_pension_recipients": pensioners[report.period],
                "source_title": report.title,
                "source_url": report.url,
                "population_source_url": MOI_DOWNLOAD_PAGE,
                "pension_recipients_source_url": pension_source_urls[report.period],
            }
        )

        if exact_balance is not None and reported_scale is not None:
            # The summary is printed only to NT$10,000, so its discrepancy from
            # the exact table total must be less than one reporting unit.
            if abs(exact_balance - reported_scale) >= 10_000:
                raise RuntimeError(
                    f"{report.period}: summary/table mismatch: "
                    f"{reported_scale} versus {exact_balance}"
                )

    missing = [
        row["date"] for row in rows
        if row["fund_level_ntd"] is None
    ]
    if missing:
        print("Missing exact balance for:", ", ".join(missing))

    row_periods = [str(row["date"]) for row in rows]
    expected_periods = [f"{year:04d}-{month:02d}" for year, month in periods]
    if row_periods != expected_periods:
        raise RuntimeError("Fund reports do not form a complete monthly sequence")

    output = Path(args.output)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
