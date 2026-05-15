#!/bin/bash
# run_experiment.sh - Batch run MLP energy comparison experiments with temperature stabilization
# Usage: bash run_experiment.sh [--dry-run] [--max-epochs 1000] [--target-r2 0.86] [--temp-threshold 50] [--stable-duration 5] [--max-wait 300]

set -e

# -------------------- 配置 Python 命令 --------------------
PYTHON_CMD="${PYTHON_CMD:-/home/js/miniconda3/bin/conda run -n python_cource python}"

# -------------------- 解析可选参数 --------------------
DRY_RUN=false
MAX_EPOCHS=1000
TARGET_R2=0.86
LEARNING_RATE=0.001
IDLE_DURATION=10.0
TEMP_THRESHOLD=50          # 目标温度（摄氏度）
STABLE_DURATION=5          # 稳定持续时间（秒）
MAX_WAIT=300               # 最大等待时间（秒）

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --max-epochs) MAX_EPOCHS="$2"; shift 2 ;;
        --target-r2) TARGET_R2="$2"; shift 2 ;;
        --learning-rate) LEARNING_RATE="$2"; shift 2 ;;
        --idle-duration) IDLE_DURATION="$2"; shift 2 ;;
        --temp-threshold) TEMP_THRESHOLD="$2"; shift 2 ;;
        --stable-duration) STABLE_DURATION="$2"; shift 2 ;;
        --max-wait) MAX_WAIT="$2"; shift 2 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

# -------------------- 温度检测函数 --------------------
# 获取当前 CPU 温度（摄氏度）
get_cpu_temp() {
    local temp_file=""
    # 查找 x86_pkg_temp 或 cpu-thermal zone
    for zone in /sys/class/thermal/thermal_zone*/temp; do
        local type_file="${zone%/temp}/type"
        if [ -f "$type_file" ] && grep -q -E "x86_pkg_temp|cpu-thermal" "$type_file" 2>/dev/null; then
            temp_file="$zone"
            break
        fi
    done
    if [ -z "$temp_file" ]; then
        # fallback: 第一个 thermal zone
        temp_file="/sys/class/thermal/thermal_zone0/temp"
    fi
    if [ -f "$temp_file" ]; then
        local temp_raw=$(cat "$temp_file" 2>/dev/null)
        if [ -n "$temp_raw" ]; then
            echo "scale=1; $temp_raw / 1000" | bc
            return 0
        fi
    fi
    echo "0"
    return 1
}

# 等待温度稳定
wait_for_temp_stable() {
    local target_temp=$1
    local stable_dur=$2
    local max_wait=$3
    local start_time=$(date +%s)
    local stable_start=0
    local current_temp=0

    echo "Waiting for CPU temperature to drop below ${target_temp}°C (max wait ${max_wait}s)..."
    while true; do
        current_temp=$(get_cpu_temp)
        if (( $(echo "$current_temp <= $target_temp" | bc -l) )); then
            if [ $stable_start -eq 0 ]; then
                stable_start=$(date +%s)
                echo "  Temperature reached $current_temp°C, stabilizing for ${stable_dur}s..."
            elif (( $(date +%s) - stable_start >= stable_dur )); then
                echo "  Temperature stabilized at $current_temp°C after $(($(date +%s) - start_time)) seconds."
                return 0
            fi
        else
            stable_start=0
        fi
        if (( $(date +%s) - start_time >= max_wait )); then
            echo "  Warning: Max wait time exceeded, current temp $current_temp°C > $target_temp°C"
            return 1
        fi
        sleep 2
    done
}

# -------------------- 检查 RAPL 接口可读性 --------------------
RAPL_PATH="/sys/class/powercap/intel-rapl:0/energy_uj"
if [ ! -r "$RAPL_PATH" ]; then
    echo "Error: RAPL file $RAPL_PATH is not readable."
    echo "Please run the following command to grant read permission:"
    echo "  sudo chmod a+r $RAPL_PATH"
    exit 1
fi

# -------------------- 路径设置 --------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_PATH="data/processed/housing_encoded.csv"
TRAIN_SCRIPT="$PROJECT_ROOT/src/train_mlp.py"

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "Error: Training script not found at $TRAIN_SCRIPT"
    exit 1
fi

if [ ! -f "$PROJECT_ROOT/$DATA_PATH" ]; then
    echo "Error: Data file not found at $PROJECT_ROOT/$DATA_PATH"
    exit 1
fi

# -------------------- 定义10种结构 --------------------
STRUCTURES=(
    "50"          # 1
    "100"         # 2
    "200"         # 3
    "50,50"       # 4
    "100,100"     # 5
    "200,200"     # 6
    "50,100,50"   # 7
    "100,50,100"  # 8
    "200,100,50"  # 9
    "50,200,50"   # 10
)

# -------------------- 实验日志 --------------------
LOG_FILE="$SCRIPT_DIR/experiment_$(date +%Y%m%d_%H%M%S).log"
echo "Experiment log will be saved to: $LOG_FILE"

START_TIME=$(date +%s)
echo "========== Experiment started: $(date) ==========" | tee -a "$LOG_FILE"
echo "Project root: $PROJECT_ROOT" | tee -a "$LOG_FILE"
echo "Python command: $PYTHON_CMD" | tee -a "$LOG_FILE"
echo "Max epochs: $MAX_EPOCHS, Target R2: $TARGET_R2, Learning rate: $LEARNING_RATE" | tee -a "$LOG_FILE"
echo "Temperature threshold: ${TEMP_THRESHOLD}°C, stable duration: ${STABLE_DURATION}s, max wait: ${MAX_WAIT}s" | tee -a "$LOG_FILE"

TOTAL_RUNS=$(( ${#STRUCTURES[@]} * 5 ))
CURRENT_RUN=0

# -------------------- 主循环 --------------------
for idx in "${!STRUCTURES[@]}"; do
    struct="${STRUCTURES[$idx]}"
    exp_num=$(printf "%02d" $((idx+1)))
    dir_name="exp_${exp_num}_${struct//,/_}"
    exp_dir="$SCRIPT_DIR/$dir_name"
    mkdir -p "$exp_dir"

    echo "========================================" | tee -a "$LOG_FILE"
    echo "Structure $exp_num/${#STRUCTURES[@]}: $struct" | tee -a "$LOG_FILE"
    echo "Output root: $exp_dir" | tee -a "$LOG_FILE"

    for run in {1..5}; do
        CURRENT_RUN=$((CURRENT_RUN + 1))
        run_dir="$exp_dir/run$run"
        mkdir -p "$run_dir"

        # 等待温度稳定（每个 run 开始前）
        wait_for_temp_stable "$TEMP_THRESHOLD" "$STABLE_DURATION" "$MAX_WAIT" | tee -a "$LOG_FILE"

        # 生成随机种子
        seed=$((idx*100 + run*7 + 42))

        echo "[$CURRENT_RUN/$TOTAL_RUNS] Running: $struct, run=$run, seed=$seed" | tee -a "$LOG_FILE"

        if [ "$DRY_RUN" = true ]; then
            echo "  (DRY RUN) Would execute:" | tee -a "$LOG_FILE"
            echo "  $PYTHON_CMD $TRAIN_SCRIPT --data $DATA_PATH --hidden-layers $struct --max-epochs $MAX_EPOCHS --target-r2 $TARGET_R2 --learning-rate $LEARNING_RATE --idle-duration $IDLE_DURATION --seed $seed --output-dir $run_dir" | tee -a "$LOG_FILE"
        else
            $PYTHON_CMD "$TRAIN_SCRIPT" \
                --data "$DATA_PATH" \
                --hidden-layers "$struct" \
                --max-epochs "$MAX_EPOCHS" \
                --target-r2 "$TARGET_R2" \
                --learning-rate "$LEARNING_RATE" \
                --idle-duration "$IDLE_DURATION" \
                --seed "$seed" \
                --output-dir "$run_dir" \
                2>&1 | tee -a "$LOG_FILE"

            if [ ${PIPESTATUS[0]} -ne 0 ]; then
                echo "Warning: Run failed (structure $struct, run $run)" | tee -a "$LOG_FILE"
            fi
        fi

        echo "  Cooling down for 5 seconds..." | tee -a "$LOG_FILE"
        sleep 5
    done
done

# -------------------- 完成 --------------------
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "========== All experiments completed ==========" | tee -a "$LOG_FILE"
echo "Total time: $((DURATION / 60)) min $((DURATION % 60)) sec" | tee -a "$LOG_FILE"

# 汇总
SUMMARY_SCRIPT="$SCRIPT_DIR/generate_summary.py"
if [ -f "$SUMMARY_SCRIPT" ]; then
    echo "Generating summary..." | tee -a "$LOG_FILE"
    $PYTHON_CMD "$SUMMARY_SCRIPT" "$SCRIPT_DIR" | tee -a "$LOG_FILE"
else
    echo "Note: Summary script not found at $SUMMARY_SCRIPT. You can manually run: $PYTHON_CMD generate_summary.py $SCRIPT_DIR" | tee -a "$LOG_FILE"
fi

echo "Experiment log saved to: $LOG_FILE"