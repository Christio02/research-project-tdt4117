"""Cranfield corpus parsing into the shared document dict shape.

`docs` shape (Assignment 1):
{doc_id: {"id": int, "title": str, "abstract": str}}
`terms` is added afterwards by tokenize.add_terms_to_docs.
"""

from __future__ import annotations

from pathlib import Path

# data/cran.all.1400 at the project root, resolved relative to this file
# so it works regardless of the current working directory.
CRAN_PATH = Path(__file__).resolve().parents[2] / "data" / "cran.all.1400"


def parse_cranfield(path: Path | str = CRAN_PATH) -> dict[int, dict]:
    docs = {}
    current_id = None
    current_field = None
    buffers = {"T": [], "A": [], "B": [], "W": []}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(".I "):
                if current_id is not None:
                    docs[current_id] = {
                        "id": current_id,
                        "title": " ".join(buffers["T"]).strip(),
                        "abstract": " ".join(buffers["W"]).strip(),
                    }
                current_id = int(line.split()[1])
                buffers = {k: [] for k in buffers}
                current_field = None
            elif line.startswith("."):
                tag = line[1:].strip()
                current_field = tag if tag in buffers else None
            else:
                if current_field is not None:
                    buffers[current_field].append(line)
    if current_id is not None:
        docs[current_id] = {
            "id": current_id,
            "title": " ".join(buffers["T"]).strip(),
            "abstract": " ".join(buffers["W"]).strip(),
        }
    print(f"Parsed {len(docs)} documents.")
    return docs
