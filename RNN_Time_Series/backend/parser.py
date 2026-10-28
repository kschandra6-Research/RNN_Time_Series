"""
NeuroCast: File Parsing & Ingestion Module
Concept & Design by K.S.Chandra 2026
"""

import os
import csv
import json
from typing import Dict, Any, List

PANDAS_AVAILABLE = False
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


def inspect_file_columns(filepath: str) -> Dict[str, Any]:
    """
    Detect available columns and sample numeric series from CSV, JSON, TXT, or Excel.
    Works natively with standard library.
    """
    ext = os.path.splitext(filepath)[1].lower()
    columns_info: Dict[str, List[float]] = {}

    if ext in [".xlsx", ".xls"]:
        if not PANDAS_AVAILABLE:
            raise ValueError("Reading Excel files (.xlsx/.xls) requires 'pandas'. Use .csv, .json, or .txt for zero-dependency parsing.")
        try:
            df = pd.read_excel(filepath)
            for col in df.columns:
                num_series = pd.to_numeric(df[col], errors='coerce').dropna().tolist()
                columns_info[str(col)] = num_series
            return columns_info
        except Exception as e:
            raise ValueError(f"Error reading Excel file: {e}")

    if ext == ".json":
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
            if isinstance(data, list) and all(isinstance(x, (int, float)) for x in data):
                columns_info["values"] = [float(x) for x in data]
                return columns_info
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                keys = list(data[0].keys())
                for k in keys:
                    vals = []
                    for row in data:
                        if isinstance(row, dict) and k in row:
                            try:
                                vals.append(float(row[k]))
                            except (ValueError, TypeError):
                                pass
                    if vals:
                        columns_info[k] = vals
                return columns_info
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, list):
                        vals = []
                        for item in v:
                            try:
                                vals.append(float(item))
                            except (ValueError, TypeError):
                                pass
                        if vals:
                            columns_info[k] = vals
                return columns_info
        except Exception as e:
            raise ValueError(f"Error parsing JSON: {e}")

    # CSV/TSV/TXT parser
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
    for enc in encodings:
        try:
            with open(filepath, "r", encoding=enc, errors="replace") as f:
                lines = [l.strip() for l in f if l.strip()]
            if not lines:
                continue

            first_chunk = "\n".join(lines[:10])
            delimiter = None
            try:
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(first_chunk, delimiters=",;\t|")
                delimiter = dialect.delimiter
            except Exception:
                if "\t" in first_chunk:
                    delimiter = "\t"
                elif ";" in first_chunk:
                    delimiter = ";"
                elif "," in first_chunk:
                    delimiter = ","

            if delimiter:
                reader = list(csv.reader(lines, delimiter=delimiter))
            else:
                reader = [[item for item in line.split() if item] for line in lines]

            if not reader:
                continue

            header = reader[0]
            has_header = False
            for token in header:
                try:
                    float(token)
                except ValueError:
                    has_header = True
                    break

            if has_header and len(reader) > 1:
                cols: Dict[str, List[float]] = {h.strip(): [] for h in header if h.strip()}
                for row in reader[1:]:
                    for h, val in zip(header, row):
                        h = h.strip()
                        if h in cols:
                            try:
                                cols[h].append(float(val.strip()))
                            except ValueError:
                                pass
                for h, vals in cols.items():
                    if vals:
                        columns_info[h] = vals
                if columns_info:
                    return columns_info

            tokens: List[float] = []
            for row in reader:
                for item in row:
                    for sub in item.replace(",", " ").split():
                        try:
                            tokens.append(float(sub))
                        except ValueError:
                            pass
            if len(tokens) >= 2:
                columns_info["series"] = tokens
                return columns_info

        except Exception:
            continue

    if not columns_info:
        raise ValueError("No numeric data columns could be extracted from this file.")
    return columns_info


def parse_raw_text_data(text_content: str, filename_hint: str = "") -> Dict[str, List[float]]:
    """
    Parse in-memory uploaded text content (CSV, JSON, TXT) directly from HTTP payload.
    """
    text_content = text_content.strip()
    if not text_content:
        raise ValueError("Provided file content is empty.")

    # Check JSON
    if text_content.startswith(("[", "{")):
        try:
            data = json.loads(text_content)
            cols: Dict[str, List[float]] = {}
            if isinstance(data, list) and all(isinstance(x, (int, float)) for x in data):
                return {"values": [float(x) for x in data]}
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                for k in data[0].keys():
                    vals = []
                    for row in data:
                        if isinstance(row, dict) and k in row:
                            try:
                                vals.append(float(row[k]))
                            except (ValueError, TypeError):
                                pass
                    if vals:
                        cols[k] = vals
                if cols:
                    return cols
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, list):
                        vals = []
                        for item in v:
                            try:
                                vals.append(float(item))
                            except (ValueError, TypeError):
                                pass
                        if vals:
                            cols[k] = vals
                if cols:
                    return cols
        except Exception:
            pass

    # Parse CSV / Lines
    lines = [l.strip() for l in text_content.splitlines() if l.strip()]
    if not lines:
        raise ValueError("No text lines found.")

    first_chunk = "\n".join(lines[:10])
    delimiter = None
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(first_chunk, delimiters=",;\t|")
        delimiter = dialect.delimiter
    except Exception:
        if "\t" in first_chunk:
            delimiter = "\t"
        elif ";" in first_chunk:
            delimiter = ";"
        elif "," in first_chunk:
            delimiter = ","

    if delimiter:
        reader = list(csv.reader(lines, delimiter=delimiter))
    else:
        reader = [[item for item in line.split() if item] for line in lines]

    if reader:
        header = reader[0]
        has_header = any(not is_float(token) for token in header)
        if has_header and len(reader) > 1:
            cols = {h.strip(): [] for h in header if h.strip()}
            for row in reader[1:]:
                for h, val in zip(header, row):
                    h = h.strip()
                    if h in cols and is_float(val.strip()):
                        cols[h].append(float(val.strip()))
            valid_cols = {k: v for k, v in cols.items() if len(v) >= 2}
            if valid_cols:
                return valid_cols

        # Fallback numeric stream
        tokens = []
        for row in reader:
            for item in row:
                for sub in item.replace(",", " ").split():
                    if is_float(sub):
                        tokens.append(float(sub))
        if len(tokens) >= 2:
            return {"series": tokens}

    raise ValueError("Unable to extract numeric time series from content.")


def is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def find_workspace_data_files() -> List[str]:
    """Scan script directory and current working directory for compatible time series files."""
    candidates = []
    seen = set()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    dirs = [project_root, script_dir]
    cwd = os.getcwd()
    if cwd not in dirs and not any(part.lower() in cwd.lower() for part in ["program files", "programs", "appdata"]):
        dirs.append(cwd)

    extensions = {".csv", ".json", ".txt", ".tsv", ".xlsx", ".xls"}

    for d in dirs:
        if os.path.isdir(d):
            try:
                for entry in sorted(os.listdir(d)):
                    ext = os.path.splitext(entry)[1].lower()
                    if ext in extensions and not entry.startswith("."):
                        lower_name = entry.lower()
                        if any(skip in lower_name for skip in ["requirement", "license", "setup", "readme"]):
                            continue
                        full = os.path.join(d, entry)
                        if full not in seen and os.path.isfile(full):
                            try:
                                cols = inspect_file_columns(full)
                                if cols:
                                    seen.add(full)
                                    candidates.append(full)
                            except Exception:
                                pass
            except Exception:
                pass
    return candidates
