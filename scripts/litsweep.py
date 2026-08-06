#!/usr/bin/env python3
"""litsweep - broad-spectrum, reproducible literature sweep.

Queries OPEN scholarly APIs (no keys, no scraping):

  * OpenAlex      - ~250M works; indexes ACM, IEEE, Springer; searches title+abstract
  * Crossref      - DOI metadata from nearly every publisher (ACM, IEEE, ACM DL)
  * DBLP          - excellent coverage of CS conferences (PEARC, SC, ICPP, HPDC)
  * arXiv         - preprints (metadata only)

Why not scrape Google Scholar / ACM DL / IEEE Xplore:
  they block automated access (Cloudflare, CAPTCHA), full text sits behind logins,
  and their ToS forbid it. The APIs above are designed for this, are citable, and
  make the review REPRODUCIBLE - which the paper needs.

Usage:
    python scripts/litsweep.py --email you@example.com
    python scripts/litsweep.py --email you@example.com --queries queries.txt --out sweep/
    python scripts/litsweep.py --email you@example.com --since 2023

The email joins OpenAlex/Crossref's polite pool: higher rate limits. Use your own.

Output:
    sweep/results.csv        all deduplicated records
    sweep/results.md         readable table, sorted by year
    sweep/by_query.json      which query found what (traceability)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

USER_AGENT = "litsweep/1.0 (academic literature review; mailto:{email})"
TIMEOUT = 30
SLEEP = 0.5  # politeness toward the APIs: do not lower this


# Default queries: the Anvil sweep. Replace with your own.
DEFAULT_QUERIES = [
    # the core: operational artifacts
    "LLM SLURM job script generation",
    "large language model batch script scheduler",
    "LLM HPC user support assistant evaluation",
    "job script correctness benchmark scheduler",
    # containers
    "LLM Apptainer Singularity definition file generation",
    "LLM Dockerfile generation build success benchmark",
    # methodological neighbours
    "execution-based benchmark infrastructure as code LLM",
    "infrastructure as code LLM benchmark Terraform",
    "executable dataset automated program repair",
    # diagnosis
    "LLM root cause analysis HPC job failure",
    "supercomputer job failure diagnosis dataset",
    # HPC context
    "HPC LLM code generation benchmark",
    "retrieval augmented generation HPC documentation",
]


@dataclass
class Record:
    title: str
    year: int | None = None
    authors: str = ""
    venue: str = ""
    doi: str = ""
    url: str = ""
    abstract: str = ""
    source: str = ""              # openalex | crossref | dblp | arxiv
    queries: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        """Dedup key: DOI when present, otherwise the normalised title."""
        if self.doi:
            return self.doi.lower().strip()
        return re.sub(r"[^a-z0-9]", "", self.title.lower())[:80]


def _get(url: str, email: str) -> bytes | None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT.format(email=email)})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 429:                       # rate limited: back off and retry
                time.sleep(2 ** attempt + 1)
                continue
            print(f"  ! HTTP {e.code} su {url[:70]}", file=sys.stderr)
            return None
        except Exception as e:  # noqa: BLE001
            print(f"  ! {type(e).__name__} su {url[:70]}", file=sys.stderr)
            time.sleep(1)
    return None


def _reconstruct_abstract(inverted: dict | None) -> str:
    """OpenAlex ships abstracts as an inverted index."""
    if not inverted:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted.items():
        positions.extend((i, word) for i in idxs)
    return " ".join(w for _, w in sorted(positions))[:1200]


# ---------------------------------------------------------------- sources
def search_openalex(query: str, email: str, since: int, limit: int) -> list[Record]:
    params = urllib.parse.urlencode({
        "search": query,
        "filter": f"from_publication_date:{since}-01-01",
        "per-page": min(limit, 50),
        "mailto": email,
    })
    raw = _get(f"https://api.openalex.org/works?{params}", email)
    if not raw:
        return []
    out = []
    for w in json.loads(raw).get("results", []):
        loc = (w.get("primary_location") or {}).get("source") or {}
        out.append(Record(
            title=(w.get("title") or "").strip(),
            year=w.get("publication_year"),
            authors="; ".join(
                a["author"]["display_name"] for a in (w.get("authorships") or [])[:5]
            ),
            venue=loc.get("display_name", "") or "",
            doi=(w.get("doi") or "").replace("https://doi.org/", ""),
            url=w.get("id", ""),
            abstract=_reconstruct_abstract(w.get("abstract_inverted_index")),
            source="openalex",
        ))
    return out


def search_crossref(query: str, email: str, since: int, limit: int) -> list[Record]:
    params = urllib.parse.urlencode({
        "query.bibliographic": query,
        "filter": f"from-pub-date:{since}-01-01,type:proceedings-article",
        "rows": min(limit, 50),
        "mailto": email,
    })
    raw = _get(f"https://api.crossref.org/works?{params}", email)
    if not raw:
        return []
    out = []
    for w in json.loads(raw).get("message", {}).get("items", []):
        titles = w.get("title") or []
        if not titles:
            continue
        date = (w.get("published") or {}).get("date-parts", [[None]])[0]
        out.append(Record(
            title=titles[0].strip(),
            year=date[0] if date else None,
            authors="; ".join(
                f"{a.get('given','')} {a.get('family','')}".strip()
                for a in (w.get("author") or [])[:5]
            ),
            venue=(w.get("container-title") or [""])[0],
            doi=w.get("DOI", ""),
            url=w.get("URL", ""),
            abstract=re.sub(r"<[^>]+>", "", w.get("abstract", ""))[:1200],
            source="crossref",
        ))
    return out


def search_dblp(query: str, email: str, since: int, limit: int) -> list[Record]:
    params = urllib.parse.urlencode({"q": query, "format": "json", "h": min(limit, 50)})
    raw = _get(f"https://dblp.org/search/publ/api?{params}", email)
    if not raw:
        return []
    hits = json.loads(raw).get("result", {}).get("hits", {}).get("hit", [])
    out = []
    for h in hits:
        info = h.get("info", {})
        year = int(info["year"]) if info.get("year", "").isdigit() else None
        if year and year < since:
            continue
        auth = info.get("authors", {}).get("author", [])
        if isinstance(auth, dict):
            auth = [auth]
        out.append(Record(
            title=info.get("title", "").rstrip("."),
            year=year,
            authors="; ".join(a.get("text", "") for a in auth[:5]),
            venue=info.get("venue", ""),
            doi=info.get("doi", ""),
            url=info.get("ee", "") or info.get("url", ""),
            source="dblp",
        ))
    return out


def search_arxiv(query: str, email: str, since: int, limit: int) -> list[Record]:
    params = urllib.parse.urlencode({
        "search_query": f"all:{query}",
        "max_results": min(limit, 50),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    })
    raw = _get(f"http://export.arxiv.org/api/query?{params}", email)
    if not raw:
        return []
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(raw)
    out = []
    for e in root.findall("a:entry", ns):
        pub = e.findtext("a:published", "", ns)
        year = int(pub[:4]) if pub[:4].isdigit() else None
        if year and year < since:
            continue
        out.append(Record(
            title=" ".join((e.findtext("a:title", "", ns) or "").split()),
            year=year,
            authors="; ".join(
                a.findtext("a:name", "", ns) for a in e.findall("a:author", ns)[:5]
            ),
            venue="arXiv",
            url=e.findtext("a:id", "", ns),
            abstract=" ".join((e.findtext("a:summary", "", ns) or "").split())[:1200],
            source="arxiv",
        ))
    return out


SOURCES = {
    "openalex": search_openalex,
    "crossref": search_crossref,
    "dblp": search_dblp,
    "arxiv": search_arxiv,
}


# ---------------------------------------------------------------- pipeline
def sweep(queries: list[str], email: str, since: int, limit: int,
          sources: list[str]) -> tuple[list[Record], dict[str, list[str]]]:
    merged: dict[str, Record] = {}
    by_query: dict[str, list[str]] = defaultdict(list)

    for q in queries:
        print(f"\n» {q}")
        for name in sources:
            recs = SOURCES[name](q, email, since, limit)
            print(f"    {name:<10} {len(recs):>3} results")
            for r in recs:
                if not r.title:
                    continue
                if r.key in merged:
                    if q not in merged[r.key].queries:
                        merged[r.key].queries.append(q)
                    # prefer the record that carries an abstract
                    if not merged[r.key].abstract and r.abstract:
                        merged[r.key].abstract = r.abstract
                else:
                    r.queries = [q]
                    merged[r.key] = r
                by_query[q].append(r.title)
            time.sleep(SLEEP)

    return list(merged.values()), dict(by_query)


def write_outputs(records: list[Record], by_query: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    records.sort(key=lambda r: (-(r.year or 0), r.title.lower()))

    with open(out / "results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(asdict(records[0]).keys()) if records else ["title"])
        w.writeheader()
        for r in records:
            d = asdict(r)
            d["queries"] = " | ".join(d["queries"])
            w.writerow(d)

    with open(out / "results.md", "w", encoding="utf-8") as fh:
        fh.write(f"# Literature sweep - {len(records)} unique records\n\n")
        fh.write("Sorted by descending year. `n_query` = how many queries surfaced it "
                 "(a high value signals centrality).\n\n")
        fh.write("| Year | Title | Venue | n_query | Source |\n|---|---|---|---|---|\n")
        for r in records:
            t = r.title.replace("|", "/")[:110]
            link = f"[{t}]({r.url})" if r.url else t
            fh.write(f"| {r.year or '?'} | {link} | {r.venue[:32]} | "
                     f"{len(r.queries)} | {r.source} |\n")

    (out / "by_query.json").write_text(json.dumps(by_query, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    print(f"\n{len(records)} unique records -> {out}/results.md")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--email", required=True, help="for the OpenAlex/Crossref polite pool")
    p.add_argument("--queries", type=Path, help="file with one query per line")
    p.add_argument("--out", type=Path, default=Path("sweep"))
    p.add_argument("--since", type=int, default=2022, help="minimum year")
    p.add_argument("--limit", type=int, default=30, help="results per query per source")
    p.add_argument("--sources", nargs="+", default=list(SOURCES),
                   choices=list(SOURCES))
    args = p.parse_args()

    queries = (
        [line.strip() for line in args.queries.read_text(encoding="utf-8").splitlines()
         if line.strip() and not line.lstrip().startswith("#")]
        if args.queries else DEFAULT_QUERIES
    )
    print(f"{len(queries)} queries x {len(args.sources)} sources, from {args.since}")

    records, by_query = sweep(queries, args.email, args.since, args.limit, args.sources)
    if not records:
        print("No results. Check your connection.", file=sys.stderr)
        return 1
    write_outputs(records, by_query, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
