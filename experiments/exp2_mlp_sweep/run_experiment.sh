#!/usr/bin/env bash
#
# Experiment 2: MLP Architecture Comparison (6 structures)
# Each architecture uses its own best hyperparams from grid search.
# 6 structures × 5 repeats = 30 runs.
#
# Usage:
#   bash run_experiment.sh

set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$EXPERIMENT_DIR/../.." && pwd)"
TRAIN_SCRIPT="$PROJECT_ROOT/experiments/exp1_model_comparison/train.py"

CONDA_PYTHON="/home/js/miniconda3/bin/conda run -n python_cource python"
RAPL_PATH="/sys/class/powercap/intel-rapl:0/energy_uj"

TEMP_THRESHOLD=50000
TEMP_STABLE_DURATION=5
MAX_WAIT_TEMP=300

BASE_SEED=42
N_REPEATS=5
IDLE_DURATION=10
COOLDOWN_SECS=30

# ─── 6 architectures with their best hyperparams ──────────────
# format: "tag|hidden|activation|lr|alpha|batch_size"
# Best configs from grid search:
CONFIGS=(
    "mlp_250x30|250,30|relu|0.0010|0.001|64"
    "mlp_128x64x32|128,64,32|tanh|0.0005|0.001|128"
    "mlp_32x64x128|32,64,128|tanh|0.0005|0.001|128"
    "mlp_70x70x70|70,70,70|tanh|0.0005|0.001|128"
    "mlp_90x60x50x30|90,60,50,30|tanh|0.0005|0.001|64"
    "mlp_40x150x40|40,150,40|tanh|0.0005|0.001|128"
)

# ─── Colours ──────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; NC='\033[0m'
log_info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

if [ "$EUID" -ne 0 ]; then
    log_error "Must run as root."
    exit 1
fi
if [ ! -r "$RAPL_PATH" ]; then
    chmod a+r "$RAPL_PATH" 2>/dev/null || true
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
    log_warn "Temp timeout"
    return 1
}

# ─── Run one repeat ───────────────────────────────────────────
run_repeat() {
    local config=$1 seed=$2 run_id=$3
    local tag hidden act lr alpha batch

    IFS='|' read -r tag hidden act lr alpha batch <<< "$config"
    local out_dir="$EXPERIMENT_DIR/results/$tag/run_${run_id}"
    mkdir -p "$out_dir"

    log_info "  -- $tag | run ${run_id} | seed ${seed} --"
    wait_for_temp "$TEMP_THRESHOLD" "$TEMP_STABLE_DURATION" "$MAX_WAIT_TEMP" || true

    cd "$EXPERIMENT_DIR"
    if taskset -c 0 $CONDA_PYTHON "$TRAIN_SCRIPT" \
        --model mlp \
        --mlp-hidden "$hidden" \
        --mlp-activation "$act" \
        --mlp-lr "$lr" \
        --mlp-alpha "$alpha" \
        --mlp-batch-size "$batch" \
        --seed "$seed" \
        --idle-duration "$IDLE_DURATION" \
        --output-dir "$out_dir" 2>&1; then
        log_ok "  ok $tag run ${run_id}"
    else
        log_error "  FAILED $tag run ${run_id}"
    fi

    sleep "$COOLDOWN_SECS"
}

# ═══════════════════ Main ═════════════════════════════════════
TOTAL_RUNS=$(( ${#CONFIGS[@]} * N_REPEATS ))
CURRENT_RUN=0

log_info "============================================"
log_info "  Experiment 2: MLP Architecture Comparison"
log_info "  ${#CONFIGS[@]} architectures x ${N_REPEATS} repeats = ${TOTAL_RUNS} runs"
log_info "============================================"
log_info ""

# Clear previous
log_info "Clearing previous results..."
rm -rf "$EXPERIMENT_DIR/results"
mkdir -p "$EXPERIMENT_DIR/results"

# System prep
cpupower frequency-set -g performance 2>/dev/null || true
command -v powertop &>/dev/null && powertop --auto-tune 2>/dev/null || true

for config in "${CONFIGS[@]}"; do
    IFS='|' read -r tag _ _ _ _ _ <<< "$config"
    log_info ""
    log_info "  === $tag ==="
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

log_ok "=== Experiment 2 finished ==="
