"""One reproducible entry point for deterministic contracts and the fixed live series.

Every phase writes a log and a verdict. No retry, skip or failed live case is
converted into success. Live mode requires the already configured local stack.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["deterministic", "live", "full"], default="deterministic"
    )
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode != "deterministic" and not args.env_file:
        parser.error("Live checks require --env-file")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "mode": args.mode,
        "platform": sys.platform,
        "phases": [],
        "windows_exclusions": (
            ["POSIX fcntl workspace tests"] if sys.platform == "win32" else []
        ),
    }

    def phase(name, command, cwd=ROOT, env=None):
        started = time.monotonic()
        print(f"START {name}", flush=True)
        with (output / f"{name}.log").open("w", encoding="utf-8") as log:
            try:
                process = subprocess.run(
                    command,
                    cwd=cwd,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=14000 if name == "live-20" else 600,
                )
                exit_code = process.returncode
            except subprocess.TimeoutExpired:
                log.write("\nHarness phase deadline exceeded\n")
                exit_code = 124
        report["phases"].append(
            {
                "name": name,
                "exit_code": exit_code,
                "seconds": round(time.monotonic() - started, 2),
                "log": f"{name}.log",
            }
        )
        (output / "harness.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(f"{'PASS' if exit_code == 0 else 'FAIL'} {name}", flush=True)
        return exit_code == 0

    if args.mode in {"deterministic", "full"}:
        env = {
            **os.environ,
            "SERVICE_AUTH_SERVER_URL": "http://auth.test",
            "SERVICE_AUTH_REALM": "test",
            "SERVICE_AUTH_CLIENT_ID": "test",
            "SERVICE_AUTH_CLIENT_SECRET": "test-placeholder",
        }
        unit = [
            sys.executable,
            "-m",
            "pytest",
            "tests/unit",
            "-q",
            "--tb=short",
            f"--junitxml={output / 'unit.xml'}",
        ]
        if sys.platform == "win32":
            unit.append("--ignore=tests/unit/test_workspace_store.py")
        phase("unit", unit, env=env)
        phase(
            "negative-contracts",
            [
                sys.executable,
                str(SCRIPTS / "contract_probes.py"),
                "--output",
                str(output / "negative-contracts.json"),
            ],
        )
        npm = shutil.which("npm.cmd" if sys.platform == "win32" else "npm")
        if not npm:
            raise RuntimeError("npm is required for the frontend acceptance contracts")
        phase("frontend-tests", [npm, "test"], cwd=ROOT / "frontend")
        phase("frontend-build", [npm, "run", "build"], cwd=ROOT / "frontend")
    if args.mode in {"live", "full"}:
        if any(p["exit_code"] for p in report["phases"]):
            print(
                "Live series not started because deterministic acceptance failed",
                flush=True,
            )
            return 1
        phase(
            "redis-contention",
            [
                sys.executable,
                str(SCRIPTS / "store_concurrency.py"),
                "--output",
                str(output / "redis-contention.json"),
            ],
        )
        common = [
            "--env-file",
            str(args.env_file.resolve()),
            "--output",
            str(output / "live"),
        ]
        phase(
            "live-20",
            [sys.executable, str(SCRIPTS / "stability.py"), *common, "--runs", "20"],
        )
        phase(
            "live-verification",
            [sys.executable, str(SCRIPTS / "verify_stability.py"), *common],
        )
    report["passed"] = bool(report["phases"]) and all(
        p["exit_code"] == 0 for p in report["phases"]
    )
    (output / "harness.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
