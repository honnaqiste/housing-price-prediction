# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """Aggregate all metrics.json under experiment root and generate summary.csv"""

# import json
# import sys
# from pathlib import Path
# import pandas as pd


# def main():
#     if len(sys.argv) != 2:
#         print("Usage: python generate_summary.py <experiment_root>")
#         sys.exit(1)

#     root_dir = Path(sys.argv[1])
#     if not root_dir.exists():
#         print(f"Error: Directory {root_dir} does not exist")
#         sys.exit(1)

#     records = []
#     # Walk through all exp_*/run*/metrics.json
#     for metrics_path in root_dir.glob("exp_*/run*/metrics.json"):
#         # Parse path info
#         exp_dir = metrics_path.parent.parent  # e.g., exp_01_50/
#         run_dir = metrics_path.parent          # e.g., run1/
#         # Restore comma-separated structure string
#         structure = exp_dir.name.split("_", 2)[-1].replace("_", ",")
#         run_id = int(run_dir.name[3:])  # run1 -> 1

#         with open(metrics_path, "r") as f:
#             data = json.load(f)

#         record = {
#             "structure": structure,
#             "run_id": run_id,
#             "seed": data["seed"],
#             "target_r2": data["target_r2"],
#             "stopped_early": data["stopped_early"],
#             "final_epoch": data["final_epoch"],
#             "total_energy_j": data["total_energy_j"],
#             "best_val_r2": data["best_val_r2"],
#             "best_epoch": data["best_epoch"],
#             "final_val_r2": data["final_val_r2"],
#             "final_info_gain": data["final_info_gain"],
#             "test_r2": data["test_r2"],
#             "test_mse": data["test_mse"],
#             "baseline_mse": data["baseline_mse"],
#             # 新增背景功率列
#             "background_power_watts": data.get("background_power_watts", None),
#             "idle_duration": data.get("idle_duration", None),
#         }
#         records.append(record)

#     if not records:
#         print("No metrics.json files found")
#         return

#     df = pd.DataFrame(records)
#     df = df.sort_values(["structure", "run_id"])
#     output_csv = root_dir / "summary.csv"
#     df.to_csv(output_csv, index=False)
#     print(f"Summary saved to: {output_csv}")
#     print(f"Total records: {len(df)}")


# if __name__ == "__main__":
#     main()

import json
from pathlib import Path
import pandas as pd

root = Path(".")  # 当前目录应该是实验根目录

records = []
json_files = list(root.glob("exp_*/run*/metrics.json"))
print(f"Found {len(json_files)} metrics.json files")

for json_file in json_files:
    # 解析路径
    run_dir = json_file.parent      # .../run1
    exp_dir = run_dir.parent        # .../exp_01_50
    # 提取结构名称：exp_01_50 -> 50 ; exp_04_50_50 -> 50,50
    # 取 exp_dir.name 去掉 "exp_XX_" 前缀，并将下划线替换回逗号
    parts = exp_dir.name.split('_', 2)  # ['exp', '01', '50'] 或 ['exp', '04', '50_50']
    if len(parts) >= 3:
        structure = parts[2].replace('_', ',')
    else:
        structure = exp_dir.name  # fallback
    run_id = int(run_dir.name[3:])   # 'run1' -> 1

    with open(json_file) as f:
        data = json.load(f)

    records.append({
        "structure": structure,
        "run_id": run_id,
        "seed": data.get("seed"),
        "stopped_early": data.get("stopped_early"),
        "total_energy_j": data.get("total_energy_j"),
        "best_val_r2": data.get("best_val_r2"),
        "final_val_r2": data.get("final_val_r2"),
        "test_r2": data.get("test_r2"),
        "background_power_watts": data.get("background_power_watts"),
    })

if not records:
    print("No records found. Please check the directory structure.")
    print("Expected pattern: exp_*/run*/metrics.json")
    exit()

df = pd.DataFrame(records)

# 只保留前5种结构（根据你实际存在的结构名称调整）
first5_structures = ['50', '100', '200', '50,50', '100,100']
df_first5 = df[df['structure'].isin(first5_structures)]

print(f"Total runs: {len(df)}, runs in first5: {len(df_first5)}")
print(df_first5.to_string())

# 可选：保存到CSV
df_first5.to_csv("first5_summary.csv", index=False)
print("\nSaved to first5_summary.csv")