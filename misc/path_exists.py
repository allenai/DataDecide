#!/usr/bin/env python3
"""
Quick-n-dirty checkpoint verifier

Checks for every <checkpoint_location>/<revision>:
  • dir exists
  • ≥1 optimizer file     (name has “optimizer” or “opt_state”)
  • ≥1 model file         (name matches *model*.bin|safetensors etc.)

Extra cross-revision test:
  • The *set of model-file sizes* must be identical in every revision
    under the same checkpoint_location.

Exit status: 0 → OK, 1 → anything missing / mismatch
"""

import json, re, sys
from pathlib import Path

OPT_RE   = re.compile(r'(optimizer|opt_state|optim)', re.I)
MODEL_RE = re.compile(r'(pytorch_model|model|consolidated).*?\.(bin|safetensors)$', re.I)

def files_matching(pattern, folder: Path):
    return [f for f in folder.iterdir() if f.is_file() and pattern.search(f.name)]

def main(jsonl_path: str) -> None:
    ok = True

    for raw in Path(jsonl_path).read_text().splitlines():
        if not raw.strip():          # skip blanks
            continue
        rec  = json.loads(raw)
        base = Path(rec["checkpoints_location"]
                    .replace("weka://oe-eval-default", "/data/input"))

        baseline_sizes = None  # type: set[int] | None

        for rev in rec["revisions"]:
            p = base / rev
            # ────────────────── existence ──────────────────
            if not p.is_dir():
                print(f"[missing dir]                {p}")
                ok = False
                continue

            # ───────────────── optimizer present ────────────
            if not files_matching(OPT_RE, p):
                print(f"[missing optimizer]          {p}")
                ok = False

            # ───────────────── model(s) present ─────────────
            models = files_matching(MODEL_RE, p)
            if not models:
                print(f"[missing model]              {p}")
                ok = False
                continue

            # gather size signature for this revision
            sizes = {m.stat().st_size for m in models}

            # establish or compare against baseline
            if baseline_sizes is None:
                baseline_sizes = sizes
            elif sizes != baseline_sizes:
                print(f"[size mismatch across revs]  {p}  (sizes: {sorted(sizes)})")
                ok = False

    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python check_ckpts.py file.jsonl")
    main(sys.argv[1])
