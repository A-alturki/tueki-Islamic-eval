#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${ROOT_DIR}/aclpubcheck_env"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Error: ${PYTHON_BIN} not found on PATH" >&2
  exit 1
fi

if [ ! -d "${ENV_DIR}" ]; then
  echo "Creating virtual environment at ${ENV_DIR}" >&2
  "${PYTHON_BIN}" -m venv "${ENV_DIR}"
else
  echo "Reusing existing virtual environment at ${ENV_DIR}" >&2
fi

# Activate the environment
# shellcheck disable=SC1090
source "${ENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install --upgrade "git+https://github.com/acl-org/aclpubcheck"

echo "aclpubcheck environment is ready at ${ENV_DIR}" >&2
