"""Run independent remote physics seeds concurrently, preserving each log."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import monotonic

if platform.system() != "Linux":
    raise SystemExit("Run validation on remote Linux, not on the Mac")

parser = argparse.ArgumentParser()
parser.add_argument("--scope", choices=("drawer", "full"), default="drawer")
parser.add_argument("--start-seed", type=int, default=100)
parser.add_argument("--attempts", type=int, default=10)
parser.add_argument("--workers", type=int, default=2)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.attempts < 1 or args.workers < 1 or args.start_seed < 0:
    parser.error("Require positive attempts/workers and nonnegative start seed")
args.output.mkdir(parents=True, exist_ok=False)
revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def attempt(seed):
    report = args.output / f"seed-{seed}.json"
    started = monotonic()
    with (args.output / f"seed-{seed}.log").open("w") as log:
        command = [sys.executable, "-u", "scripts/validate_expert.py", "--scope", args.scope,
                   "--start-seed", str(seed), "--attempts", "1", "--timeout", "120",
                   "--report", str(report)]
        try:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=150)
        except subprocess.TimeoutExpired:
            return {"seed": seed, "success": False, "error": "Outer process timeout"}
    if not report.exists():
        return {"seed": seed, "success": False, "error": f"No report; exit {result.returncode}"}
    row = json.loads(report.read_text())[0]
    row["process_seconds"] = round(monotonic() - started, 3)
    row["returncode"] = result.returncode
    row["success"] = bool(row["success"] and result.returncode == 0)
    return row


rows = []
with ThreadPoolExecutor(max_workers=args.workers) as pool:
    pending = [pool.submit(attempt, seed) for seed in range(args.start_seed, args.start_seed + args.attempts)]
    for future in as_completed(pending):
        row = future.result()
        rows.append(row)
        print(json.dumps({k: row.get(k) for k in ("seed", "success", "elapsed_seconds", "error")}), flush=True)
        summary = {"revision": revision, "scope": args.scope, "workers": args.workers,
                   "rows": sorted(rows, key=lambda r: r["seed"])}
        (args.output / "report.json").write_text(json.dumps(summary, indent=2))
passed = sum(row["success"] for row in rows)
print(f"{passed}/{len(rows)} {args.scope} seeds passed", flush=True)
if passed != len(rows):
    raise SystemExit(1)
