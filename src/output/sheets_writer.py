"""Output writer — Google Sheets (Deliverable 1) with CSV fallback.

Produces the six required tabs: Startups, Products, Research Papers, Jobs, News,
Entity Mapping Log. Records are flattened to a stable column order so the sheet
is human-readable. If no Google service-account credentials are configured, we
write one CSV per tab to ./out/ so the pipeline is always runnable offline.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from src.core.logging import get_logger
from src.resolver.entity_resolver import MappingResult
from src.settings import get_settings

log = get_logger(__name__)

_TABS = ["Startups", "Products", "Research Papers", "Jobs", "News", "Entity Mapping Log"]


def _flatten(rec: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested envelope into dotted columns for tabular output."""
    out: dict[str, Any] = {}

    def walk(node: Any, prefix: str = "") -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{prefix}{k}.")
        elif isinstance(node, list):
            out[prefix.rstrip(".")] = "; ".join(map(str, node))
        else:
            out[prefix.rstrip(".")] = node

    walk(rec)
    return out


def _rows(records: list[dict]) -> tuple[list[str], list[list[Any]]]:
    flat = [_flatten(r) for r in records]
    headers: list[str] = []
    for f in flat:
        for k in f:
            if k not in headers:
                headers.append(k)
    rows = [[f.get(h, "") for h in headers] for f in flat]
    return headers, rows


def _mapping_rows(log_entries: list[MappingResult]) -> tuple[list[str], list[list[Any]]]:
    headers = ["raw", "canonical", "method", "score"]
    rows = [[m.raw, m.canonical, m.method, m.score] for m in log_entries]
    return headers, rows


class OutputWriter:
    def __init__(self) -> None:
        self.settings = get_settings()

    def write(self, output: dict[str, list[dict]], mapping_log: list[MappingResult]) -> None:
        tab_data: dict[str, tuple[list[str], list[list[Any]]]] = {
            "Startups": _rows(output["startups"]),
            "Products": _rows(output["products"]),
            "Research Papers": _rows(output["papers"]),
            "Jobs": _rows(output["jobs"]),
            "News": _rows(output["news"]),
            "Entity Mapping Log": _mapping_rows(mapping_log),
        }
        if self.settings.google_service_account_json and self.settings.google_sheet_id:
            self._write_sheets(tab_data)
        else:
            self._write_csv(tab_data)

    def _write_csv(self, tab_data: dict[str, tuple[list[str], list[list[Any]]]]) -> None:
        out_dir = Path("./out")
        out_dir.mkdir(parents=True, exist_ok=True)
        for tab, (headers, rows) in tab_data.items():
            path = out_dir / f"{tab.replace(' ', '_').lower()}.csv"
            with path.open("w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(headers)
                w.writerows(rows)
            log.info("csv_written", tab=tab, rows=len(rows), path=str(path))

    def _write_sheets(self, tab_data: dict[str, tuple[list[str], list[list[Any]]]]) -> None:
        import gspread
        from google.oauth2.service_account import Credentials

        creds = Credentials.from_service_account_file(
            self.settings.google_service_account_json,
            scopes=["https://www.googleapis.com/auth/spreadsheets"],
        )
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(self.settings.google_sheet_id)
        for tab, (headers, rows) in tab_data.items():
            try:
                ws = sh.worksheet(tab)
                ws.clear()
            except gspread.WorksheetNotFound:
                ws = sh.add_worksheet(title=tab, rows=max(len(rows) + 10, 100), cols=max(len(headers), 10))
            ws.update([headers, *rows], value_input_option="RAW")
            log.info("sheet_tab_written", tab=tab, rows=len(rows))
