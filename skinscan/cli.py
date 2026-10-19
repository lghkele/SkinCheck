import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import uuid

import cv2

from . import __version__
from .config import load_config, page_config
from .engine import analyze_image
from .models import DetectionResult
from .report import write_artifacts, write_html, write_json


def _read_object(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _inputs(args):
    jobs = []
    if args.manifest:
        manifest_path = Path(args.manifest).resolve()
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = value.get("screenshots") if isinstance(value, dict) else value
        if not isinstance(records, list):
            raise ValueError("Manifest must be a list or an object with a screenshots list")
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("imagePath"), str):
                raise ValueError("Each manifest entry requires imagePath")
            path = Path(record["imagePath"])
            if not path.is_absolute():
                path = manifest_path.parent / path
            jobs.append((path.resolve(), record))
    for source in args.inputs:
        path = Path(source).resolve()
        paths = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() == ".png") if path.is_dir() else [path]
        jobs.extend((p, None) for p in paths)
    if not jobs:
        raise ValueError("No screenshots supplied/found")
    if args.metadata:
        if len(jobs) != 1 or args.manifest:
            raise ValueError("--metadata requires exactly one input image and no manifest")
        jobs[0] = (jobs[0][0], _read_object(args.metadata))
    # Do not scan a previous report, nor let report output overwrite a source.
    root = Path(args.output).resolve()
    if any(root == path or root in path.parents for path, _ in jobs):
        raise ValueError("Input screenshots must be outside the output directory; choose a separate --output")
    return jobs


def _run_scan(args):
    started = time.perf_counter()
    config = load_config(args.config)
    if args.workers is not None:
        if not 1 <= args.workers <= 32:
            raise ValueError("--workers must be between 1 and 32")
        config["runtime"]["concurrency"] = args.workers
    jobs = _inputs(args)
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%S") + "Z-" + uuid.uuid4().hex[:8]
    output = Path(args.output).resolve() / run_id
    output.mkdir(parents=True, exist_ok=False)
    # Bound OpenCV's internal pools; screenshot and chain concurrency are managed here.
    cv2.setNumThreads(1)

    def process(item):
        index, (path, metadata) = item
        debug = {} if config["runtime"]["save_debug_masks"] else None
        try:
            if metadata is None:
                sidecar = path.with_suffix(".json")
                metadata = _read_object(sidecar) if sidecar.is_file() else {}
            # Page overrides may enable debug masks for a single page.
            page_id = str(metadata.get("pageId") or path.stem)
            if page_config(config, page_id)["runtime"]["save_debug_masks"]:
                debug = {}
            result, frame = analyze_image(path, config, metadata, debug)
        except Exception as exc:
            result, frame = DetectionResult(path.stem, str(path), status="error",
                                             errors=[f"{type(exc).__name__}: {exc}"]), None
        slug = re.sub(r"[^\w.-]+", "_", result.pageId, flags=re.UNICODE).strip(".")[:60] or "page"
        digest = hashlib.sha256(str(path).encode()).hexdigest()[:8]
        artifact_dir = f"{index+1:04d}-{slug}-{digest}"
        record = write_artifacts(output / artifact_dir, result, frame, debug)
        record["artifactDirectory"] = artifact_dir
        write_json(output / artifact_dir / "review.json", {
            "schemaVersion": 1, "pageId": result.pageId, "verdict": "unreviewed", "issues": {}})
        print(f"{result.status:9} {result.pageId} ({result.durationMs:.0f} ms)", flush=True)
        return record

    # map preserves manifest order and releases each image/mask when its job is finished.
    with ThreadPoolExecutor(max_workers=config["runtime"]["concurrency"]) as pool:
        records = list(pool.map(process, enumerate(jobs)))
    durations = sorted(record["durationMs"] for record in records)
    summary = {"schemaVersion": 1, "version": __version__, "runId": run_id,
               "createdAt": now.isoformat(), "policy": "strict" if args.strict else "report_only",
               "counts": dict(Counter(record["status"] for record in records)), "results": records,
               "durationMs": round((time.perf_counter()-started)*1000, 2),
               "meanImageDurationMs": round(sum(durations) / len(durations), 2),
               "p95ImageDurationMs": durations[min(len(durations)-1, int(len(durations)*0.95))]}
    write_json(output / "summary.json", summary)
    if config["runtime"]["html_report"]:
        write_html(output, summary)
    print(f"Report: {output}")
    # Input/module errors always fail; report-only never conceals execution failures.
    if any(record["errors"] for record in records):
        return 2
    if args.strict and any(record["status"] == "failed" for record in records):
        return 1
    if args.strict and any(record["status"] == "uncertain" for record in records):
        return 2
    return 0


def _review(args):
    directory = Path(args.result).resolve()
    if directory.is_file():
        directory = directory.parent
    result = _read_object(directory / "result.json")
    review_path = directory / "review.json"
    review = _read_object(review_path) if review_path.exists() else {"schemaVersion": 1, "pageId": result["pageId"], "issues": {}}
    if args.issue:
        def ids(issues):
            for issue in issues:
                yield issue["id"]
                yield from ids(issue["children"])
        if args.issue not in set(ids(result["issues"])):
            raise ValueError(f"Unknown issue ID: {args.issue}")
        target = review.setdefault("issues", {}).setdefault(args.issue, {})
    else:
        target = review
    entry = {"verdict": args.verdict, "reviewer": args.reviewer,
             "note": args.note, "updatedAt": datetime.now(timezone.utc).isoformat()}
    target.update(entry)
    review.setdefault("history", []).append({"issueId": args.issue, **entry})
    # Keep machine observations immutable; human decisions are a separate, append-history artifact.
    write_json(review_path, review)
    print(f"Review saved: {review_path}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="skinscan", description="Full-resolution PNG dark-theme checks (CPU, no OCR)")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    scan = commands.add_parser("scan", help="Analyze images/directories or a screenshot manifest")
    scan.add_argument("inputs", nargs="*")
    scan.add_argument("--manifest")
    scan.add_argument("--metadata")
    scan.add_argument("--config")
    scan.add_argument("--output", default="reports")
    scan.add_argument("--workers", type=int)
    scan.add_argument("--strict", action="store_true", help="Exit 1 for confirmed-risk findings, 2 for uncertain analysis")
    scan.set_defaults(handler=_run_scan)
    review = commands.add_parser("review", help="Save human feedback alongside a result")
    review.add_argument("result", help="Result directory or result.json path")
    review.add_argument("--issue", help="Optional issue ID; default reviews the whole page")
    review.add_argument("--verdict", required=True, choices=["confirmed", "dismissed", "unresolved"])
    review.add_argument("--reviewer", required=True)
    review.add_argument("--note", default="")
    review.set_defaults(handler=_review)
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (ValueError, TypeError, OSError, KeyError, cv2.error) as exc:
        print(f"skinscan: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
