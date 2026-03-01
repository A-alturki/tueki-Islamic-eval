#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "paper"
LATEX_DIR = PAPER_DIR / "latex"
DEFAULT_OUTPUT = PAPER_DIR / "BurhanAI_at_IslamicEval_2025_Shared_Task_camera_ready.pdf"
ENV_BIN = ROOT / "aclpubcheck_env" / "bin"
ACLPUBCHECK = ENV_BIN / "aclpubcheck"


def run_cmd(cmd, *, cwd=None, check=True, capture_output=False):
    result = subprocess.run(
        cmd,
        cwd=cwd,
        check=check,
        text=True,
        capture_output=capture_output,
    )
    return result.stdout if capture_output else None


def compile_latex(output_pdf: Path, keep_logs: bool = False):
    build_env = os.environ.copy()
    build_env["PATH"] = f"{ENV_BIN}:{build_env['PATH']}"

    if shutil.which("tectonic", path=build_env["PATH"]) is None:
        raise RuntimeError("tectonic not found on PATH; install via `brew install tectonic`")

    tmp_dir = tempfile.mkdtemp(prefix="latex_build_")
    try:
        run_cmd([
            "tectonic",
            "--outdir",
            tmp_dir,
            "acl_latex.tex",
        ], cwd=LATEX_DIR, check=True)

        generated_pdf = Path(tmp_dir) / "acl_latex.pdf"
        if not generated_pdf.exists():
            raise RuntimeError("LaTeX compilation succeeded but output PDF not found")

        shutil.copy2(generated_pdf, output_pdf)

        if keep_logs:
            for log_name in ("acl_latex.log", "acl_latex.blg", "acl_latex.aux"):
                src = Path(tmp_dir) / log_name
                if src.exists():
                    shutil.copy2(src, LATEX_DIR / log_name)
    finally:
        if not keep_logs:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def run_aclpubcheck(pdf_path: Path, paper_type: str = "long", fail_on_errors: bool = True):
    file_stub = pdf_path.stem.split("_")[0]
    cmd = [str(ACLPUBCHECK), "--paper_type", paper_type, str(pdf_path)]
    stdout = run_cmd(cmd, capture_output=True)
    print(stdout)

    error_json = ROOT / f"errors-{file_stub}.json"
    if error_json.exists():
        with error_json.open() as f:
            errors = json.load(f)
        if errors and fail_on_errors:
            raise RuntimeError(f"ACL pubcheck reported errors: {json.dumps(errors, indent=2)}")
    return stdout


def main():
    parser = argparse.ArgumentParser(description="Compile LaTeX and run aclpubcheck")
    parser.add_argument("pdf", nargs="?", default=str(DEFAULT_OUTPUT), help="Path to PDF or output location")
    parser.add_argument("--compile", action="store_true", help="Compile LaTeX before running aclpubcheck")
    parser.add_argument("--paper-type", default="long", choices=["long", "short", "demo", "other"], help="Paper type")
    parser.add_argument("--keep-logs", action="store_true", help="Keep intermediate LaTeX logs")
    parser.add_argument("--ignore-errors", action="store_true", help="Do not fail even if errors JSON exists")
    args = parser.parse_args()

    pdf_path = Path(args.pdf).resolve()

    if args.compile:
        compile_latex(pdf_path, keep_logs=args.keep_logs)

    run_aclpubcheck(pdf_path, paper_type=args.paper_type, fail_on_errors=not args.ignore_errors)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
