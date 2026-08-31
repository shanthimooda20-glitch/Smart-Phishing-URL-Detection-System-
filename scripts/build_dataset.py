"""Rebuild ``data/phishing_urls.csv`` from its public sources.

The repository already ships a ready-to-train dataset; this script exists so
the corpus is fully reproducible and auditable rather than a mystery blob.

Sources
-------
1. **Malicious/benign URL corpus** - ``incertum/cyber-matrix-ai``
   (``url_data_mega_deep_learning.csv``): ~195k full URLs labelled
   ``isMalicious`` 0/1. URLs are stored without a scheme.
2. **Popular-domain list** - ``zer0h/top-1000000-domains`` (top 100k):
   used only to top up the *benign, path-less* stratum (see below).

Why the extra source, and why structural stratification?
--------------------------------------------------------
Corpus (1) carries two collection artefacts that a naive sample bakes straight
into the model:

* 77% of its **path-less** URLs are malicious, simply because the benign half
  was harvested as full page URLs. A model trained on that learns
  "no path => bad" and flags ``example.com``.
* Its benign half is heavily **normalised** — bare hosts, almost no
  subdomains — so any subdomain looks malicious by association.

Neither is a property of phishing; both are properties of the collection
process. The builder therefore stratifies on URL *structure* (path present,
and hostname depth) and balances the two classes **inside every structural
bucket**, so the model cannot use bucket membership as a shortcut and has to
earn its verdict from the lexical evidence within the bucket. The benign
path-less bucket is topped up with real popular domains, which the corpus
barely covers.

Usage
-----
    python scripts/build_dataset.py                  # 40,000-row balanced sample
    python scripts/build_dataset.py --rows 80000
    python scripts/build_dataset.py --full           # every usable row
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import sys
import urllib.request
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "phishing_urls.csv"

CORPUS_URL = (
    "https://raw.githubusercontent.com/incertum/cyber-matrix-ai/master/"
    "Malicious-URL-Detection-Deep-Learning/data/url_data_mega_deep_learning.csv"
)
TOP_DOMAINS_URL = (
    "https://raw.githubusercontent.com/zer0h/top-1000000-domains/master/"
    "top-100000-domains"
)

#: Share of the output reserved for URLs that contain a path.
PATH_SHARE = 0.70

#: Hostname-depth buckets (number of dots in the host, 3 = "3 or more").
DEPTH_BUCKETS = (0, 1, 2, 3)
RANDOM_SEED = 42


def _download(url: str, timeout: int = 180) -> str:
    """Fetch ``url`` and return its body as text."""
    print(f"  downloading {url.rsplit('/', 1)[-1]} ...", flush=True)
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
        return response.read().decode("utf-8", errors="replace")


def load_corpus() -> pd.DataFrame:
    """Return the cleaned base corpus as ``url``/``label`` rows."""
    rows = list(csv.reader(io.StringIO(_download(CORPUS_URL))))
    frame = pd.DataFrame(rows[1:], columns=["url", "label"])
    frame["url"] = frame["url"].astype(str).str.strip()
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
    frame = frame.dropna(subset=["label"])
    frame = frame[frame["url"].str.len() >= 4]
    frame["label"] = frame["label"].astype(int)
    return frame.drop_duplicates(subset=["url"])


def load_top_domains(exclude: Iterable[str]) -> pd.DataFrame:
    """Return popular domains that do not collide with the corpus."""
    domains = [line.strip().lower() for line in _download(TOP_DOMAINS_URL).splitlines()]
    blocked = set(exclude)
    keep = [
        domain for domain in domains
        if domain and "." in domain and " " not in domain and domain not in blocked
    ]
    return pd.DataFrame({"url": keep, "label": 0})


def _sample(frame: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    """Return at most ``size`` rows drawn without replacement."""
    if size >= len(frame):
        return frame
    return frame.sample(n=size, random_state=seed)


def _allocate(capacity: dict[int, int], budget: int) -> dict[int, int]:
    """Split ``budget`` rows-per-class across the depth buckets.

    Allocation is proportional to the *square root* of each bucket's balanced
    capacity rather than to the capacity itself. Straight proportional
    sampling would leave the dominant "one dot" bucket with ~90% of the rows
    and give the model almost no subdomain examples to learn from; the square
    root keeps that bucket the largest while still giving the deeper buckets
    enough mass. Leftover budget from capped buckets is redistributed.
    """
    weights = {depth: math.sqrt(size) for depth, size in capacity.items() if size > 0}
    if not weights:
        return {}

    allocation: dict[int, int] = {}
    remaining = budget
    open_buckets = dict(weights)

    for _ in range(3):  # a couple of redistribution passes is plenty
        total_weight = sum(open_buckets.values())
        if not open_buckets or remaining <= 0 or total_weight == 0:
            break
        assigned = 0
        for depth, weight in list(open_buckets.items()):
            wanted = allocation.get(depth, 0) + int(remaining * weight / total_weight)
            granted = min(wanted, capacity[depth])
            assigned += granted - allocation.get(depth, 0)
            allocation[depth] = granted
            if granted >= capacity[depth]:
                open_buckets.pop(depth)
        remaining -= assigned
        if assigned == 0:
            break

    return {depth: size for depth, size in sorted(allocation.items()) if size > 0}


def _host_depth(url: str) -> int:
    """Return the hostname's dot count, clipped to :data:`DEPTH_BUCKETS`."""
    host = url.split("/")[0].split(":")[0].lower()
    if host.startswith("www."):
        host = host[4:]
    return min(host.count("."), DEPTH_BUCKETS[-1])


def build(rows: int | None, output: Path, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Build the structurally stratified dataset and write it to ``output``."""
    print("Loading sources:")
    corpus = load_corpus()
    malicious_hosts = {
        url.split("/")[0].split(":")[0].lower()
        for url in corpus.loc[corpus["label"] == 1, "url"]
    }
    top_domains = load_top_domains(exclude=malicious_hosts)

    corpus["structure"] = corpus["url"].str.contains("/").map({True: "path", False: "bare"})
    pool = pd.concat(
        [corpus[["url", "label", "structure"]], top_domains.assign(structure="bare")],
        ignore_index=True,
    ).drop_duplicates(subset=["url"])
    pool["depth"] = pool["url"].map(_host_depth)

    parts: list[pd.DataFrame] = []
    for structure, share in (("path", PATH_SHARE), ("bare", 1 - PATH_SHARE)):
        block = pool[pool["structure"] == structure]

        # Balanced mass available in each hostname-depth bucket.
        capacity = {
            depth: min(
                ((block["depth"] == depth) & (block["label"] == 0)).sum(),
                ((block["depth"] == depth) & (block["label"] == 1)).sum(),
            )
            for depth in DEPTH_BUCKETS
        }
        total_capacity = sum(capacity.values())
        if total_capacity == 0:
            continue

        budget = total_capacity * 2 if rows is None else int(rows * share)
        print(f"  {structure}: {total_capacity * 2:,} balanced rows available")

        for depth, per_class in _allocate(capacity, budget // 2).items():
            bucket = block[block["depth"] == depth]
            for label in (0, 1):
                drawn = _sample(
                    bucket[bucket["label"] == label][["url", "label"]],
                    per_class,
                    seed + label,
                )
                parts.append(drawn)
            print(f"    depth={depth}: {per_class:,} per class")

    dataset = (
        pd.concat(parts, ignore_index=True)
        .drop_duplicates(subset=["url"])
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output, index=False)

    print()
    print(f"Wrote {len(dataset):,} rows to {output}")
    print(f"  phishing   : {int(dataset['label'].sum()):,}")
    print(f"  legitimate : {int((dataset['label'] == 0).sum()):,}")
    return dataset


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, default=40000,
                        help="Total number of rows to write")
    parser.add_argument("--full", action="store_true",
                        help="Use every usable row instead of a sample")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args(argv)
    try:
        build(None if args.full else args.rows, args.output, args.seed)
    except (OSError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
