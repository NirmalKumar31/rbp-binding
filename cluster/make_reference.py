# Co-authored with Claude (Anthropic).
"""Generate cluster/positives_ref.tsv from an existing processed dir (run LOCALLY, once).

Writes protein -> sha256 of everything-but-split. validate.py recomputes the same hash on the
frozen cluster data to prove the re-extract reproduces v2 byte-for-byte.

    python cluster/make_reference.py ../rbp-v2/data/processed
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from validate import content_hash

proc = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "data/processed"
proteins = [l.split("\t")[0] for l in (REPO / "config/proteins.tsv").read_text().splitlines()[1:]]

out = ["protein\tsha256"]
for p in proteins:
    d = proc / p / "dataset.tsv"
    if d.exists():
        out.append(f"{p}\t{content_hash(d)}")
        print(p, "hashed")
    else:
        print(p, "-- no dataset.tsv, skipped")
(REPO / "cluster/positives_ref.tsv").write_text("\n".join(out) + "\n")
print("wrote", REPO / "cluster/positives_ref.tsv")
