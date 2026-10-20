"""Export every task of a caregapoptimization_e2e job run as readable Markdown.

Each notebook becomes evidence/run_outputs/<task_key>.md with its code cells and the
real outputs from that run (text, tables, errors), so execution evidence can be read as text.

Usage:
    python scripts/export_run_outputs.py <job_run_id> --profile <PROFILE> [--out evidence/run_outputs]
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import urllib.parse

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
MAX_TABLE_ROWS = 20
MAX_TEXT_CHARS = 6000


def cli(*args: str, profile: str) -> dict:
    out = subprocess.run(["databricks", *args, "--profile", profile, "-o", "json"],
                         check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def notebook_model(html: str) -> dict:
    blob = re.search(r"__DATABRICKS_NOTEBOOK_MODEL = '([^']+)'", html).group(1)
    return json.loads(urllib.parse.unquote(base64.b64decode(blob).decode()))


def render_table(res: dict) -> str:
    cols = [c["name"] for c in res.get("schema", [])]
    rows = res.get("data", [])
    if not cols:
        return ""
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows[:MAX_TABLE_ROWS]:
        lines.append("| " + " | ".join(str(v).replace("|", "\\|").replace("\n", " ") for v in r) + " |")
    if len(rows) > MAX_TABLE_ROWS:
        lines.append(f"\n_{len(rows) - MAX_TABLE_ROWS} more rows not shown ({len(rows)} total)._")
    return "\n".join(lines)


def render_results(res: dict | None) -> list[str]:
    if not res:
        return []
    kind = res.get("type")
    if kind == "listResults":
        return [part for r in res.get("data", []) for part in render_results(r)]
    if kind == "table":
        t = render_table(res)
        return [t] if t else []
    data = res.get("data")
    if kind in ("ansi", "text", "raw") and isinstance(data, str) and data.strip():
        text = ANSI.sub("", data).rstrip()
        if len(text) > MAX_TEXT_CHARS:
            text = "…(truncated)…\n" + text[-MAX_TEXT_CHARS:]
        return ["```\n" + text + "\n```"]
    return []


def to_markdown(task_key: str, run: dict, nb: dict) -> str:
    st = run["state"]
    md = [f"# {task_key}", "",
          f"- Notebook: `{nb.get('name')}`",
          f"- Task run id: {run['run_id']}  |  Result: {st.get('result_state')}", ""]
    for c in sorted(nb["commands"], key=lambda c: float(c.get("position", 0))):
        code = c.get("command", "").rstrip()
        if code.startswith("%md"):
            md += [code[3:].strip(), ""]
            continue
        if not code:
            continue
        lang = "sql" if code.startswith("%sql") else "python"
        md += [f"```{lang}", code, "```", ""]
        outs = render_results(c.get("results"))
        if c.get("error"):
            outs.append("```\n" + ANSI.sub("", str(c["error"]))[-MAX_TEXT_CHARS:] + "\n```")
        if outs:
            md += ["**Output:**", "", *[o + "\n" for o in outs]]
    return "\n".join(md)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--out", default="evidence/run_outputs")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    job_run = cli("jobs", "get-run", a.run_id, profile=a.profile)
    for task in sorted(job_run["tasks"], key=lambda t: t.get("start_time", 0)):
        export = cli("jobs", "export-run", str(task["run_id"]), "--views-to-export", "CODE", profile=a.profile)
        nb = notebook_model(export["views"][0]["content"])
        path = os.path.join(a.out, f"{task['task_key']}.md")
        with open(path, "w") as f:
            f.write(to_markdown(task["task_key"], task, nb))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
