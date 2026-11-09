#!/usr/bin/env bash
# Phase 6c: checks of the Docker image that need neither a GPU nor the model weights.
# This is what CI runs after `docker build`; the GPU run (scripts/docker_smoke.sh) stays a local step.
#
#   docker build -t defectlens:ci .
#   bash scripts/docker_static_checks.sh --tag defectlens:ci
#
# Options: --tag TAG (defectlens:local)  --port N (8775)
#          DOCKER=docker.exe  use another docker client (e.g. Docker Desktop's, from WSL without integration)
#
# What it checks: the image's size (printed, not limited), torch and TensorRT are not in it, the process is not
# root, a HEALTHCHECK is declared; then it starts the image WITHOUT a GPU and WITHOUT weights, read-only root,
# all capabilities dropped, and checks that /health answers on CPUExecutionProvider and says why, that it
# carries the frozen-thresholds file of this checkout (same SHA-256), that /products lists five products
# that are all unavailable, that /demo/ loads, and that /inspect answers 503 for a product whose model file is
# missing and 404 for an unknown product. It does not run a model: that needs the weights and is the smoke test.
set -euo pipefail
cd "$(dirname "$0")/.."

DOCKER=${DOCKER:-docker}
TAG=defectlens:local PORT=8775
while [ $# -gt 0 ]; do
  case "$1" in
    --tag) TAG=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done

WORK=$(mktemp -d)
NAME=defectlens-static-$$
PASSED=0
cleanup() { "$DOCKER" rm -f "$NAME" >/dev/null 2>&1 || true; rm -rf "$WORK"; }
trap cleanup EXIT

fail() {
  echo "FAIL: $*" >&2
  "$DOCKER" logs --tail 40 "$NAME" >&2 2>&1 || true
  exit 1
}
ok() { PASSED=$((PASSED + 1)); echo "ok   $1"; }
# pj FILE EXPR: evaluate a python expression with the JSON file loaded as j
pj() { python3 -c 'import json,sys; j=json.load(open(sys.argv[1])); print(eval(sys.argv[2]))' "$1" "$2"; }

command -v curl >/dev/null || fail "curl is needed"
"$DOCKER" image inspect "$TAG" >/dev/null 2>&1 || fail "image $TAG not found; run: docker build -t $TAG ."

SIZE=$("$DOCKER" image inspect "$TAG" --format '{{.Size}}' | tr -d '\r')
echo "image $TAG: $SIZE bytes ($(python3 -c "print(round($SIZE / 2**30, 2))") GiB by 'docker image inspect .Size')"

# ---- properties of the image ----------------------------------------------------------------------
"$DOCKER" run --rm --entrypoint python "$TAG" -c \
  "import importlib.util as u, sys; sys.exit(u.find_spec('torch') is not None)" \
  || fail "torch is installed in the image"
ok "image: torch is not installed"

[ -z "$("$DOCKER" run --rm --entrypoint sh "$TAG" -c \
  'find / -xdev \( -name "libnvinfer*" -o -name "tensorrt*" \) 2>/dev/null | head -1' | tr -d '\r')" ] \
  || fail "a TensorRT library is in the image"
ok "image: no TensorRT library"

UID_IN=$("$DOCKER" run --rm --entrypoint id "$TAG" -u | tr -d '\r')
[ "$UID_IN" != 0 ] || fail "the image runs as root"
ok "image: runs as uid $UID_IN"

HC=$("$DOCKER" image inspect "$TAG" --format '{{if .Config.Healthcheck}}yes{{end}}' | tr -d '\r')
[ "$HC" = yes ] || fail "the image declares no HEALTHCHECK"
ok "image: HEALTHCHECK declared"

# ---- the service, without GPU and without weights -------------------------------------------------------
"$DOCKER" run -d --name "$NAME" -p "127.0.0.1:$PORT:8000" --read-only --tmpfs /tmp --cap-drop=ALL \
  --security-opt no-new-privileges:true "$TAG" >/dev/null
BASE=http://127.0.0.1:$PORT
up=0
for ((i = 0; i < 120; i++)); do
  if curl -fsS "$BASE/health" -o "$WORK/health.json" 2>/dev/null; then up=1; break; fi
  sleep 1
done
[ "$up" = 1 ] || fail "/health did not answer within 120 s"

[ "$(pj "$WORK/health.json" "j['provider']")" = CPUExecutionProvider ] \
  || fail "no GPU was given but /health says $(pj "$WORK/health.json" "j['provider']")"
[ -n "$(pj "$WORK/health.json" "j['fallback_reason']")" ] || fail "/health gives no fallback_reason on CPU"
ok "service: /health answers on CPUExecutionProvider and says why ($(pj "$WORK/health.json" "j['status']"), no weights mounted)"

WANT=$(sha256sum reports/frozen_thresholds.json | cut -d' ' -f1)
[ "$(pj "$WORK/health.json" "j['frozen_file_sha256']")" = "$WANT" ] \
  || fail "the image's frozen thresholds ($(pj "$WORK/health.json" "j['frozen_file_sha256']")) differ from this checkout ($WANT)"
ok "service: frozen thresholds in the image are this checkout's file (sha256 ${WANT:0:12})"

curl -fsS "$BASE/products" -o "$WORK/products.json" || fail "/products does not answer"
[ "$(pj "$WORK/products.json" "len(j)")" = 5 ] || fail "/products does not list five products"
[ "$(pj "$WORK/products.json" "any(p['available'] for p in j)")" = False ] \
  || fail "a product is available although no weights are mounted"
ok "service: /products lists 5 products, none available without weights"

curl -fsS "$BASE/demo/" -o /dev/null || fail "the demo page /demo/ does not load"
ok "service: demo page loads"

code=$(curl -sS -o /dev/null -w '%{http_code}' -F image=@tests/fixtures/sample.jpg "$BASE/inspect?product=steel")
[ "$code" = 503 ] || fail "/inspect steel without weights returned $code, wanted 503"
code=$(curl -sS -o /dev/null -w '%{http_code}' -F image=@tests/fixtures/sample.jpg "$BASE/inspect?product=nope")
[ "$code" = 404 ] || fail "/inspect with an unknown product returned $code, wanted 404"
ok "service: /inspect answers 503 for a missing model and 404 for an unknown product"

echo "static checks passed: $PASSED checks (no GPU, no weights, no model was run)"
