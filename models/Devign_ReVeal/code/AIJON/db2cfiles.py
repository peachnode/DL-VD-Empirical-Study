#!/usr/bin/env python3
import argparse
import re
import json
import sqlite3
from pathlib import Path
import pandas as pd

LABEL_MAP = {
    "NOT_HELPFUL": 0,
    "SATURATED": 1,
    "UNREACHED": 2,
    "BUILD_ERROR": 3,
    "SUCCESS": 4,
    "WRONG_FORMAT": 5,
    "INSERT_ERROR": 6,
}

def extract_function_name(code: str) -> str:
    m = re.search(r'\b([A-Za-z_]\w*)\s*\([^)]*\)\s*{?', code)
    return m.group(1) if m else "unknown"

def sanitize_name(s: str) -> str:
    s = re.sub(r"[^\w.-]", "_", s)
    return s[:200] or "row"

def parse_annotations(text: str):
    if not isinstance(text, str) or not text.strip():
        return {}
    lines = text.splitlines()
    i = 0
    ann = {}
    if i < len(lines) and lines[i].strip().lower() in {"c", "c++"}:
        i += 1
    while i < len(lines):
        if lines[i].strip().upper().startswith("LINE "):
            m = re.match(r"\s*LINE\s+(\d+)\s*$", lines[i], re.IGNORECASE)
            i += 1
            if not m:
                continue
            ln = int(m.group(1))
            snippet_lines = []
            while i < len(lines) and not re.match(r"\s*LINE\s+\d+\s*$", lines[i], re.IGNORECASE):
                snippet_lines.append(lines[i])
                i += 1
            while snippet_lines and snippet_lines[0].strip() == "":
                snippet_lines.pop(0)
            snippet = ("\n".join(snippet_lines)).rstrip("\n")
            if snippet:
                ann.setdefault(ln, []).append(snippet)
        else:
            i += 1
    return ann

def restore_from_line_numbers(code: str) -> str:
    """
    Rebuilds newlines based on line numbers in sequence (1,2,3,...).
    Example: 'foo 1 bar 2 baz' → 'foo\nbar\nbaz'
    """
    # Match sequences like ' 12 ' that are likely line markers
    parts = re.split(r'\s+\d+\s+', code)
    return "\n".join(parts)

def strip_line_numbers(code: str) -> str:
    return re.sub(r'^[ \t]*\d+[ \t]*(?:[:|.])?[ \t]*', '', code, flags=re.MULTILINE)

def apply_annotations(code: str, annotations_text: str) -> str:
    ann = parse_annotations(annotations_text)
    # Rebuild lines first so LINE n matches real lines
    code = restore_from_line_numbers(code)

    if not ann:
        return strip_line_numbers(code)

    src = code.splitlines()
    out = []
    for idx, line in enumerate(src, start=1):
        out.append(line)
        # Support both LINE n (1-based) and LINE n (stored as n-1)
        if idx in ann or (idx - 1) in ann:
            for key in (idx, idx - 1):
                if key in ann:
                    for snippet in reversed(ann[key]):
                        out.extend(snippet.splitlines())

    merged = "\n".join(out) + ("\n" if code.endswith("\n") else "")
    # Final cleanup
    return strip_line_numbers(restore_from_line_numbers(merged))


SQL = """
    SELECT
        s.name AS score,
        f.code AS code,
        e.annotation AS annotation
    FROM main.examples e
    LEFT JOIN main.scores s    ON s.id = e.score_id
    LEFT JOIN main.functions f ON f.id = e.function_id
    ORDER BY e.id
"""

def read_db(path, limit):
    con = sqlite3.connect(path)
    try:
        if limit > 0:
            return pd.read_sql_query(SQL + " LIMIT ?", con, params=[limit])
        return pd.read_sql_query(SQL, con)
    finally:
        con.close()

def main():
    ap = argparse.ArgumentParser(description="Export C files from multiple DB splits.")
    ap.add_argument("--train", help="Path to SQLite DB for TRAIN split")
    ap.add_argument("--val", help="Path to SQLite DB for VAL split")
    ap.add_argument("--test", help="Path to SQLite DB for TEST split")
    ap.add_argument("-n", "--max-rows", type=int, default=0,
                    help="Per-split row cap; 0 means all")
    args = ap.parse_args()

    splits = []
    if args.train: splits.append(("train", read_db(args.train, args.max_rows)))
    if args.val:   splits.append(("val",   read_db(args.val,   args.max_rows)))
    if args.test:  splits.append(("test",  read_db(args.test,  args.max_rows)))
    if not splits:
        raise SystemExit("Provide at least one of --train/--val/--test")

    total_rows = sum(len(df) for _, df in splits)

    basedir = Path("data") / f"labels_{total_rows}"
    outdir = basedir / "raw_code"
    ggnn_dir = basedir / "ggnn_input"
    outdir.mkdir(parents=True, exist_ok=True)
    ggnn_dir.mkdir(parents=True, exist_ok=True)

    split_files = {
        "train": (basedir / "split_train.txt").open("w", encoding="utf-8"),
        "val":   (basedir / "split_val.txt").open("w", encoding="utf-8"),
        "test":  (basedir / "split_test.txt").open("w", encoding="utf-8"),
    }
    try:
        file_entries = []
        counter = 0
        counts = {}

        for split_name, df in splits:
            for _, row in df.iterrows():
                counter += 1
                score = str(row["score"]).strip().upper()
                label = LABEL_MAP.get(score, -1)
                code = str(row["code"])

                if "annotation" in df.columns and isinstance(row["annotation"], str):
                    code = apply_annotations(code, row["annotation"])
                else:
                    code = strip_line_numbers(restore_from_line_numbers(code))

                # final cleanup
                code = strip_line_numbers(restore_from_line_numbers(code))

                base = sanitize_name(extract_function_name(code))
                counts[base] = counts.get(base, 0) + 1
                filename = f"{counter}_{base}_{counts[base]}_{label}.c"

                (outdir / filename).write_text(code, encoding="utf-8")
                file_entries.append({"file_name": filename})
                split_files[split_name].write(f"{filename}\n")

        (ggnn_dir / "cfg_full_text_files.json").write_text(
            json.dumps(file_entries, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        (basedir / "label_map.json").write_text(
            json.dumps(LABEL_MAP, indent=2), encoding="utf-8"
        )

    finally:
        for f in split_files.values():
            f.close()

if __name__ == "__main__":
    main()
