#!/usr/bin/env bash
#
# Experiment 1: Cross-model energy efficiency comparison
# - Prepares system (cpupower, powertop)
# - Pins to CPU core 0 (taskset)
# - Waits for CPU temperature ≤50°C before each run
# - Runs: LinearRegression, RandomForest, HistGB, MLP(256,128,64),
#         MLP(128,64,32), MLP(128,64)
# - 5 repeats each with different seeds
#
# Usage:
#   bash prepare_measurement.sh   (first, optional)
#   bash run_experiment.sh
#   bash run_experiment.sh --repeats 3

set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$EXPERIMENT_DIR/../.." && pwd)"
TRAIN_SCRIPT="$EXPERIMENT_DIR/train.py"

CONDA_PYTHON="/home/js/miniconda3/bin/conda run -n python_cource python"
RAPL_PATH="/sys/class/powercap/intel-rapl:0/energy_uj"

# Temperature
TEMP_THRESHOLD=50000
TEMP_STABLE_DURATION=5
MAX_WAIT_TEMP=300

# Experiment params
BASE_SEED=42
N_REPEATS=5
IDLE_DURATION=10
COOLDOWN_SECS=30

# ─── Define each model run config ─────────────────────────────
# Each entry: "output_dir_name|model_type|extra_args"
CONFIGS=(
    "linear|linear|"
    "random_forest|random_forest|"
    "histgb|histgb|"
    "mlp_128_64_32|mlp|--mlp-hidden 128,64,32 --mlp-activation tanh --mlp-lr 0.0005 --mlp-alpha 0.001 --mlp-batch-size 128"
    "mlp_128_64|mlp|--mlp-hidden 128,64 --mlp-activation tanh --mlp-lr 0.0005 --mlp-alpha 0.001 --mlp-batch-size 128"
)

# ─── Colours ──────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; NC='\033[0m'
log_info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ─── Parse args ───────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --repeats) N_REPEATS="$2"; shift 2 ;;
        *) log_error "Unknown: $1"; exit 1 ;;
    esac
done

# ─── Checks ───────────────────────────────────────────────────
if [ "$EUID" -ne 0 ]; then
    log_error "Must run as root."
    exit 1
fi
if [ ! -r "$RAPL_PATH" ]; then
    chmod a+r "$RAPL_PATH" 2>/dev/null || log_warn "RAPL not readable"
fi
if [ ! -f "$TRAIN_SCRIPT" ]; then
    log_error "train.py not found at $TRAIN_SCRIPT"
    exit 1
fi

# ─── Temperature ──────────────────────────────────────────────
get_cpu_temp() {
    for zone in /sys/class/thermal/thermal_zone*/temp; do
        local tf="${zone%/temp}/type"
        if [ -f "$tf" ] && grep -q -E "x86_pkg_temp|cpu-thermal|TCPU" "$tf" 2>/dev/null; then
            cat "$zone" 2>/dev/null || echo "0"
            return
        fi
    done
    cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo "0"
}

wait_for_temp() {
    local threshold=${1:-$TEMP_THRESHOLD}
    local stable_dur=${2:-$TEMP_STABLE_DURATION}
    local max_wait=${3:-$MAX_WAIT_TEMP}
    local stable_start=-1 elapsed=0 temp

    log_info "Cooling CPU to <= $((threshold/1000))C ..."
    while [ "$elapsed" -lt "$max_wait" ]; do
        temp=$(get_cpu_temp)
        if [ "$temp" -le "$threshold" ] && [ "$temp" -gt 0 ]; then
            [ "$stable_start" -eq -1 ] && stable_start=$elapsed
            if [ "$((elapsed - stable_start))" -ge "$stable_dur" ]; then
                log_ok "CPU $((temp/1000))C stable"
                return 0
            fi
        else
            stable_start=-1
        fi
        sleep 1; elapsed=$((elapsed + 1))
    done
    log_warn "Temp timeout: $((temp/1000))C"
    return 1
}

# ─── Run one model, one repeat ────────────────────────────────
run_repeat() {
    local config=$1
    local seed=$2
    local run_id=$3
    local out_tag model_type extra_args

    # Parse config
    IFS='|' read -r out_tag model_type extra_args <<< "$config"

    local out_dir="$EXPERIMENT_DIR/results/$out_tag/run_${run_id}"

    log_info ""
    log_info "  -- $out_tag | run ${run_id} | seed ${seed} --"
    log_info "  Output: $out_dir"
    log_info ""

    mkdir -p "$out_dir"
    wait_for_temp "$TEMP_THRESHOLD" "$TEMP_STABLE_DURATION" "$MAX_WAIT_TEMP" || true

    cd "$EXPERIMENT_DIR"
    if taskset -c 0 $CONDA_PYTHON "$TRAIN_SCRIPT" \
        --model "$model_type" \
        --seed "$seed" \
        --idle-duration "$IDLE_DURATION" \
        --output-dir "$out_dir" \
        $extra_args 2>&1; then
        log_ok "  ok $out_tag run ${run_id} done"
    else
        log_error "  FAILED $out_tag run ${run_id}"
    fi

    log_info "  Cooling ${COOLDOWN_SECS}s ..."
    sleep "$COOLDOWN_SECS"
}

# ═══════════════════ Main ═════════════════════════════════════
TOTAL_RUNS=$(( ${#CONFIGS[@]} * N_REPEATS ))
CURRENT_RUN=0

log_info "============================================"
log_info "  Experiment 1: Model Energy Efficiency"
log_info "  ${#CONFIGS[@]} models x ${N_REPEATS} repeats = ${TOTAL_RUNS} runs"
log_info "============================================"
log_info ""

# System prep
log_info "Clearing previous results..."
rm -rf "$EXPERIMENT_DIR/results"
mkdir -p "$EXPERIMENT_DIR/results"
log_info "Setting CPU governor to performance..."
cpupower frequency-set -g performance 2>/dev/null || true
if command -v powertop &>/dev/null; then
    powertop --auto-tune 2>/dev/null || true
fi

for config in "${CONFIGS[@]}"; do
    IFS='|' read -r out_tag _ _ <<< "$config"
    log_info ""
    log_info "  === $out_tag ==="

    for i in $(seq 0 $((N_REPEATS - 1))); do
        CURRENT_RUN=$((CURRENT_RUN + 1))
        seed=$((BASE_SEED + i))
        log_info "[$CURRENT_RUN/$TOTAL_RUNS]"
        run_repeat "$config" "$seed" "$((i + 1))"
    done
done

# Analysis
log_info ""
log_info "=== All done. Analysing... ==="
cd "$EXPERIMENT_DIR"
$CONDA_PYTHON analyze.py --results-dir "$EXPERIMENT_DIR/results" --output-dir "$EXPERIMENT_DIR"

log_info ""
log_ok "=== Experiment 1 finished ==="
