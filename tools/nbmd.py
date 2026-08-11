"""Extract and reinject markdown cells of Jupyter notebooks.

Used by the PT-BR translation effort. The point is surgical safety: only
markdown cells are ever touched. Code cells, outputs, execution counts and
notebook metadata come back byte-identical.

Usage:
    python tools/nbmd.py extract <notebook.ipynb> <out.md>
    python tools/nbmd.py inject  <notebook.ipynb> <in.md>
    python tools/nbmd.py verify  <notebook.ipynb> [--against <ref.ipynb>]
"""

import argparse
import json
import subprocess
import sys

# Chosen because it cannot appear in prose or in markdown syntax.
DELIM = "<<<@@CELL"


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _detect_newline(path):
    """Line ending used by the file on disk.

    The notebooks in this repository are stored with CRLF. Writing them back
    with LF would rewrite every line of the file, burying the translated cells
    in thousands of lines of whitespace noise.
    """
    with open(path, "rb") as fh:
        raw = fh.read()
    return "\r\n" if b"\r\n" in raw else "\n"


def _dump(nb, path, newline="\n"):
    # nbformat writes a trailing newline; matching it keeps diffs to the cells.
    with open(path, "w", encoding="utf-8", newline=newline) as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
        fh.write("\n")


def _as_text(source):
    return source if isinstance(source, str) else "".join(source)


def _as_lines(text):
    """Split back into the line list nbformat expects (newline kept per line)."""
    lines = text.splitlines(keepends=True)
    return lines if lines else [""]


def extract(nb_path, out_path):
    nb = _load(nb_path)
    chunks = []
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "markdown":
            continue
        chunks.append(f"{DELIM} {idx} >>>\n{_as_text(cell['source'])}")

    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(chunks))
    print(f"extracted {len(chunks)} markdown cells -> {out_path}")


def inject(nb_path, md_path):
    nb = _load(nb_path)
    newline = _detect_newline(nb_path)
    with open(md_path, encoding="utf-8") as fh:
        raw = fh.read()

    if not raw.startswith(DELIM):
        sys.exit(f"error: {md_path} does not start with a cell delimiter")

    injected = 0
    for block in raw.split(DELIM)[1:]:
        header, _, body = block.partition(">>>\n")
        idx = int(header.strip())
        cell = nb["cells"][idx]
        if cell["cell_type"] != "markdown":
            sys.exit(f"error: cell {idx} in {nb_path} is not markdown")
        # Strip the single newline the join added between chunks.
        cell["source"] = _as_lines(body[:-1] if body.endswith("\n") else body)
        injected += 1

    _dump(nb, nb_path, newline)
    print(f"injected {injected} markdown cells -> {nb_path}")


def verify(nb_path, ref):
    """Confirm nothing but markdown cell text changed against a git ref."""
    show = subprocess.run(
        ["git", "show", f"{ref}:{nb_path}"],
        capture_output=True, text=True, encoding="utf-8",
    )
    if show.returncode != 0:
        sys.exit(f"error: cannot read {nb_path} at {ref}")

    old = json.loads(show.stdout)
    new = _load(nb_path)

    problems = []
    if len(old["cells"]) != len(new["cells"]):
        problems.append(
            f"cell count changed: {len(old['cells'])} -> {len(new['cells'])}"
        )
    if old.get("metadata") != new.get("metadata"):
        problems.append("notebook metadata changed")
    if (old.get("nbformat"), old.get("nbformat_minor")) != (
        new.get("nbformat"), new.get("nbformat_minor")
    ):
        problems.append("nbformat version changed")

    translated = 0
    for idx, (a, b) in enumerate(zip(old["cells"], new["cells"])):
        if a["cell_type"] != b["cell_type"]:
            problems.append(f"cell {idx}: type changed")
            continue
        if a["cell_type"] == "markdown":
            if _as_text(a["source"]) != _as_text(b["source"]):
                translated += 1
            continue
        # Code and raw cells must be untouched in every respect.
        if _as_text(a["source"]) != _as_text(b["source"]):
            problems.append(f"cell {idx}: CODE SOURCE CHANGED")
        if a.get("outputs") != b.get("outputs"):
            problems.append(f"cell {idx}: outputs changed")
        if a.get("execution_count") != b.get("execution_count"):
            problems.append(f"cell {idx}: execution_count changed")

    if problems:
        print(f"FAILED {nb_path}")
        for p in problems:
            print(f"  - {p}")
        return 1

    print(f"ok {nb_path} ({translated} markdown cells translated, code intact)")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ex = sub.add_parser("extract")
    p_ex.add_argument("notebook")
    p_ex.add_argument("out")

    p_in = sub.add_parser("inject")
    p_in.add_argument("notebook")
    p_in.add_argument("md")

    p_ve = sub.add_parser("verify")
    p_ve.add_argument("notebook", nargs="+")
    p_ve.add_argument("--against", default="HEAD")

    args = parser.parse_args()

    if args.cmd == "extract":
        extract(args.notebook, args.out)
    elif args.cmd == "inject":
        inject(args.notebook, args.md)
    else:
        failures = sum(verify(nb, args.against) for nb in args.notebook)
        sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
