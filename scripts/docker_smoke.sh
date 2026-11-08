#!/usr/bin/env bash
# Phase 6b smoke test of the Docker image: /health, /inspect for steel and one VisA product,
# the provider that really ran, and a few properties of the image itself.
#
#   docker build -t defectlens:local .
#   bash scripts/docker_smoke.sh                       # starts its own containers (needs --gpus)
#   bash scripts/docker_smoke.sh --url http://localhost:8000   # a service that is already running
#
# Options: --tag TAG (defectlens:local)  --image FILE (tests/fixtures/sample.jpg)
#          --visa PRODUCT (visa_pcb1)    --port N (8765)    --models DIR (./models)
#          --allow-cpu  accept CPUExecutionProvider (default: the GPU run must use CUDA)
#          --no-record  do not write reports/docker_smoke.json
#
# The container run checks, besides the API: torch is not in the image, no TensorRT library is in
# it, the process is not root, the weights mount is read-only, and a second container started
# WITHOUT --gpus reports CPUExecutionProvider and says why (so /health is not claiming a GPU that
# is not there). The synthetic default image is not a NEU-DET / VisA image: this checks that the
# service works, not that its answers are good. Exit status 0 only if every check passed.
set -euo pipefail
cd "$(dirname "$0")/.."

TAG=defectlens:local IMAGE=tests/fixtures/sample.jpg VISA=visa_pcb1 PORT=8765 URL=""
MODELS="$PWD/models" ALLOW_CPU=0 RECORD=1
while [ $# -gt 0 ]; do
  case "$1" in
    --tag) TAG=$2; shift 2 ;;
    --image) IMAGE=$2; shift 2 ;;
    --visa) VISA=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    --models) MODELS=$(cd "$2" && pwd); shift 2 ;;
    --url) URL=${2%/}; shift 2 ;;
    --allow-cpu) ALLOW_CPU=1; shift ;;
    --no-record) RECORD=0; shift ;;
    -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done

WORK=$(mktemp -d)
NAMES=()
PASSED=()
cleanup() {
  for n in "${NAMES[@]:-}"; do [ -n "$n" ] && docker rm -f "$n" >/dev/null 2>&1 || true; done
  rm -rf "$WORK"
}
trap cleanup EXIT

fail() {
  echo "FAIL: $*" >&2
  for n in "${NAMES[@]:-}"; do
    [ -n "$n" ] && { echo "--- last log lines of $n ---" >&2; docker logs --tail 25 "$n" >&2 2>&1 || true; }
  done
  exit 1
}
ok() { PASSED+=("$1"); echo "ok   $1"; }
# pj FILE EXPR: evaluate a python expression with the JSON file loaded as j
pj() { python3 -c 'import json,sys; j=json.load(open(sys.argv[1])); print(eval(sys.argv[2]))' "$1" "$2"; }

[ -f "$IMAGE" ] || fail "image $IMAGE not found"
command -v curl >/dev/null || fail "curl is needed"

wait_health() { # base-url seconds
  local i
  for ((i = 0; i < $2; i++)); do
    curl -fsS "$1/health" -o "$WORK/h.json" 2>/dev/null && return 0
    sleep 1
  done
  return 1
}

inspect() { # base-url product outfile -> HTTP code in $CODE
  CODE=$(curl -sS -o "$3" -w '%{http_code}' -F "image=@$IMAGE" "$1/inspect?product=$2&heatmap=true")
}

check_service() { # base-url label expect-provider
  local base=$1 label=$2 want=$3 p
  wait_health "$base" 120 || fail "$label: /health did not answer within 120 s"
  cp "$WORK/h.json" "$WORK/$label.health.json"
  [ "$(pj "$WORK/h.json" "j['status']")" = ok ] || fail "$label: /health status is not ok: $(cat "$WORK/h.json")"
  [ "$(pj "$WORK/h.json" "j['provider']")" = "$want" ] \
    || fail "$label: /health provider is $(pj "$WORK/h.json" "j['provider']"), wanted $want"
  ok "$label: /health ok, provider $want"
  for p in steel "$VISA"; do
    [ "$(pj "$WORK/h.json" "'$p' in j['available_products']")" = True ] \
      || fail "$label: product $p is not available: $(pj "$WORK/h.json" "j['unavailable_products']")"
    inspect "$base" "$p" "$WORK/$label.$p.json"
    [ "$CODE" = 200 ] || fail "$label: /inspect $p returned HTTP $CODE: $(head -c 300 "$WORK/$label.$p.json")"
    [ "$(pj "$WORK/$label.$p.json" "j['provider']")" = "$want" ] \
      || fail "$label: /inspect $p ran on $(pj "$WORK/$label.$p.json" "j['provider']"), wanted $want"
    [ "$(pj "$WORK/$label.$p.json" "type(j['flagged']).__name__ + str(len(j['operating_point']['frozen_file_sha256']))")" = bool64 ] \
      || fail "$label: /inspect $p answer has an unexpected shape"
    ok "$label: /inspect $p 200 on $want ($(pj "$WORK/$label.$p.json" "'flagged' if j['flagged'] else 'not flagged'"), $(pj "$WORK/$label.$p.json" "round(j['timing']['total_ms'])") ms with model load $(pj "$WORK/$label.$p.json" "round(j['timing']['load_ms'])") ms)"
  done
  [ "$(pj "$WORK/$label.$VISA.json" "j['anomaly']['heatmap_png_base64'] is not None")" = True ] \
    || fail "$label: the $VISA answer has no heatmap"
  curl -fsS "$base/demo/" -o /dev/null || fail "$label: the demo page /demo/ does not load"
  ok "$label: heatmap present, demo page loads"
}

SIZE_BYTES=null IMAGE_ID="" FS_MIB=null
if [ -n "$URL" ]; then
  check_service "$URL" service "$([ "$ALLOW_CPU" = 1 ] && echo CPUExecutionProvider || echo CUDAExecutionProvider)"
else
  command -v docker >/dev/null || fail "docker is needed"
  docker image inspect "$TAG" >/dev/null 2>&1 || fail "image $TAG not found; run: docker build -t $TAG ."
  [ -f "$MODELS/onnx/faster_rcnn.onnx" ] || fail "no weights at $MODELS/onnx (mount your models directory with --models)"
  SIZE_BYTES=$(docker image inspect "$TAG" --format '{{.Size}}')
  IMAGE_ID=$(docker image inspect "$TAG" --format '{{.Id}}')
  # what the files in the image add up to; Docker's own figure (.Size, `docker images`) can be larger
  FS_MIB=$(docker run --rm --entrypoint sh "$TAG" -c 'du -sxm / 2>/dev/null | tail -1' | cut -f1)
  echo "image $TAG: docker reports $SIZE_BYTES bytes ($(python3 -c "print(round($SIZE_BYTES/2**30, 2))") GiB)," \
       "files inside add up to $FS_MIB MiB, $IMAGE_ID"

  run_args=(-d -p "127.0.0.1:$PORT:8000" -v "$MODELS:/app/models:ro"
            --read-only --tmpfs /tmp --cap-drop=ALL --security-opt no-new-privileges:true)
  GPU=defectlens-smoke-gpu-$$
  NAMES+=("$GPU")
  docker run --name "$GPU" --gpus=all "${run_args[@]}" "$TAG" >/dev/null
  GPU_WANT=$([ "$ALLOW_CPU" = 1 ] && echo CPUExecutionProvider || echo CUDAExecutionProvider)
  check_service "http://127.0.0.1:$PORT" gpu "$GPU_WANT"

  # properties of the image
  docker exec "$GPU" python -c "import importlib.util as u, sys; sys.exit(u.find_spec('torch') is not None)" \
    || fail "torch is installed in the image"
  ok "image: torch is not installed"
  [ -z "$(docker exec "$GPU" sh -c 'find / -xdev \( -name "libnvinfer*" -o -name "tensorrt*" \) 2>/dev/null | head -1')" ] \
    || fail "a TensorRT library is in the image"
  ok "image: no TensorRT library"
  [ "$(docker exec "$GPU" id -u)" != 0 ] || fail "the service runs as root"
  ok "image: runs as uid $(docker exec "$GPU" id -u)"
  if docker exec "$GPU" sh -c 'touch /app/models/.write_test' 2>/dev/null; then
    fail "/app/models is writable inside the container"
  fi
  ok "image: weights mount is read-only, root filesystem is read-only"

  # the same image without a GPU: /health must say CPU and why, and inspect must still work
  CPU=defectlens-smoke-cpu-$$
  NAMES+=("$CPU")
  cpu_args=("${run_args[@]/127.0.0.1:$PORT:8000/127.0.0.1:$((PORT + 1)):8000}")
  docker run --name "$CPU" "${cpu_args[@]}" "$TAG" >/dev/null
  wait_health "http://127.0.0.1:$((PORT + 1))" 120 || fail "no-GPU container: /health did not answer"
  cp "$WORK/h.json" "$WORK/nogpu.health.json"
  [ "$(pj "$WORK/h.json" "j['provider']")" = CPUExecutionProvider ] \
    || fail "no-GPU container claims provider $(pj "$WORK/h.json" "j['provider']")"
  [ -n "$(pj "$WORK/h.json" "j['fallback_reason']")" ] || fail "no-GPU container gives no fallback_reason"
  inspect "http://127.0.0.1:$((PORT + 1))" "$VISA" "$WORK/nogpu.$VISA.json"
  [ "$CODE" = 200 ] || fail "no-GPU container: /inspect $VISA returned HTTP $CODE"
  [ "$(pj "$WORK/nogpu.$VISA.json" "j['provider']")" = CPUExecutionProvider ] \
    || fail "no-GPU container: /inspect ran on $(pj "$WORK/nogpu.$VISA.json" "j['provider']")"
  ok "no GPU: /health says CPUExecutionProvider with a reason, /inspect $VISA answers on CPU"
fi

if [ "$RECORD" = 1 ]; then
  PASSED_JOINED=$(printf '%s\n' "${PASSED[@]}")
  python3 - "$WORK" "$TAG" "$SIZE_BYTES" "$IMAGE_ID" "$IMAGE" "$VISA" "$PASSED_JOINED" "$([ -n "$URL" ] && echo url || echo container)" "$FS_MIB" <<'PY'
import datetime, json, pathlib, sys

work, tag, size, image_id, image, visa, passed, mode, fs_mib = sys.argv[1:10]
w = pathlib.Path(work)


def load(name):
    p = w / name
    return json.loads(p.read_text()) if p.exists() else None


def brief(a):
    return None if a is None else {
        "provider": a["provider"], "flagged": a["flagged"], "model": a["model"],
        "tf32": a["tf32"], "total_ms": round(a["timing"]["total_ms"], 1),
        "model_load_ms": round(a["timing"]["load_ms"], 1),
        "inference_ms": round(a["timing"]["inference_ms"], 1),
    }


def service(label):
    h = load(f"{label}.health.json")
    if h is None:
        return None
    return {
        "health": {k: h[k] for k in ("status", "provider", "fallback_reason", "onnxruntime")},
        "inspect": {p: brief(load(f"{label}.{p}.json")) for p in ("steel", visa)},
    }


out = {
    "date": datetime.date.today().isoformat(),
    "mode": mode,
    "image_tag": tag,
    "image_id": image_id or None,
    "image_size_bytes": None if size == "null" else int(size),
    "image_size_gib": None if size == "null" else round(int(size) / 2**30, 3),
    "image_size_note": "image_size_bytes is `docker image inspect .Size`; image_filesystem_mib is "
                       "`du -sx /` inside a container of the image (the files themselves)",
    "image_filesystem_mib": None if fs_mib == "null" else int(fs_mib),
    "test_image": image,
    "note": "first call per product includes the model load; synthetic test image, so scores are not meaningful",
    "gpu_container": service("gpu") or service("service"),
    "container_without_gpu": (lambda h: None if h is None else {
        "health": {k: h[k] for k in ("status", "provider", "fallback_reason")},
        "inspect": {visa: brief(load(f"nogpu.{visa}.json"))},
    })(load("nogpu.health.json")),
    "checks_passed": passed.splitlines(),
}
pathlib.Path("reports/docker_smoke.json").write_text(json.dumps(out, indent=2) + "\n")
print("wrote reports/docker_smoke.json")
PY
fi
echo "smoke test passed: ${#PASSED[@]} checks"
