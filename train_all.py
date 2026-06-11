#!/usr/bin/env python3
"""
全角色训练控制器

按顺序执行所有角色的 build_db → V1 → V2 → V3 → V4，
实时输出到控制台并同时写入 logs/train_<timestamp>.log。

用法:
  python train_all.py                  # 全部角色，全部步骤
  python train_all.py --chars ironclad silent
  python train_all.py --steps db v3 v4
  python train_all.py --chars watcher --steps v3 v4
"""

import argparse
import datetime
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 步骤定义
# ---------------------------------------------------------------------------

# 各角色支持的步骤（None = 跳过）
CHARACTERS = ["ironclad", "silent", "defect", "watcher"]

STEPS = {
    "db": {
        "ironclad": ["python", "-m", "ironclad_advisor.build_db"],
        "silent":   ["python", "-m", "silent_advisor.build_db"],
        "defect":   ["python", "-m", "defect_advisor.build_db"],
        "watcher":  ["python", "-m", "watcher_advisor.build_db"],
    },
    "v1": {
        "ironclad": ["python", "-m", "ironclad_advisor.ml_advisor", "train"],
        "silent":   None,   # 暂无 V1
        "defect":   None,   # 暂无 V1
        "watcher":  ["python", "-m", "watcher_advisor.ml_advisor", "train"],
    },
    "v2": {
        "ironclad": ["python", "-m", "ironclad_advisor.ml_advisor_v2", "train"],
        "silent":   ["python", "-m", "silent_advisor.ml_advisor_v2", "train"],
        "defect":   ["python", "-m", "defect_advisor.ml_advisor_v2", "train"],
        "watcher":  ["python", "-m", "watcher_advisor.ml_advisor_v2", "train"],
    },
    "v3": {
        "ironclad": ["python", "-m", "ironclad_advisor.ml_advisor_v3", "train"],
        "silent":   ["python", "-m", "silent_advisor.ml_advisor_v3", "train"],
        "defect":   ["python", "-m", "defect_advisor.ml_advisor_v3", "train"],
        "watcher":  ["python", "-m", "watcher_advisor.ml_advisor_v3", "train"],
    },
    "v4": {
        "ironclad": ["python", "-m", "ironclad_advisor.ml_advisor_v4", "train"],
        "silent":   ["python", "-m", "silent_advisor.ml_advisor_v4", "train"],
        "defect":   ["python", "-m", "defect_advisor.ml_advisor_v4", "train"],
        "watcher":  ["python", "-m", "watcher_advisor.ml_advisor_v4", "train"],
    },
}

STEP_ORDER = ["db", "v1", "v2", "v3", "v4"]

# ---------------------------------------------------------------------------
# 日志工具
# ---------------------------------------------------------------------------

class Tee:
    """同时写入文件和 stdout。"""
    def __init__(self, log_path: Path):
        self.log = open(log_path, "w", encoding="utf-8", buffering=1)
        self.stdout = sys.stdout

    def write(self, text: str):
        self.stdout.write(text)
        self.log.write(text)

    def flush(self):
        self.stdout.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def banner(tee: Tee, text: str, char: str = "="):
    line = char * 66
    tee.write(f"\n{line}\n  {text}\n{line}\n")


def timestamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# 执行单个步骤
# ---------------------------------------------------------------------------

def run_step(tee: Tee, char: str, step: str) -> bool:
    """运行一个步骤，实时流式输出，返回是否成功。"""
    cmd = STEPS[step][char]
    if cmd is None:
        tee.write(f"  [{timestamp()}] 跳过 {char} {step.upper()}（未实现）\n")
        return True

    tee.write(f"\n  [{timestamp()}] 开始 {char} {step.upper()}\n")
    tee.write(f"  命令: {' '.join(cmd)}\n\n")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    for line in proc.stdout:
        tee.write("    " + line)

    proc.wait()

    if proc.returncode == 0:
        tee.write(f"\n  [{timestamp()}] {char} {step.upper()} 完成 ✓\n")
        return True
    else:
        tee.write(f"\n  [{timestamp()}] {char} {step.upper()} 失败 ✗ (exit {proc.returncode})\n")
        return False


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="全角色训练控制器")
    parser.add_argument(
        "--chars", nargs="+",
        choices=CHARACTERS, default=CHARACTERS,
        help="要训练的角色（默认全部）",
    )
    parser.add_argument(
        "--steps", nargs="+",
        choices=STEP_ORDER, default=STEP_ORDER,
        help="要执行的步骤（默认 db v1 v2 v3 v4）",
    )
    parser.add_argument(
        "--skip-errors", action="store_true",
        help="某步骤失败时继续执行后续步骤（默认失败即停止）",
    )
    args = parser.parse_args()

    # 按定义顺序排序（即使用户乱序输入）
    chars = [c for c in CHARACTERS if c in args.chars]
    steps = [s for s in STEP_ORDER if s in args.steps]

    # 准备日志目录
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"train_{ts}.log"

    tee = Tee(log_path)

    banner(tee, f"杀戮尖塔 AI 训练控制器  {timestamp()}")
    tee.write(f"  角色: {', '.join(chars)}\n")
    tee.write(f"  步骤: {', '.join(steps)}\n")
    tee.write(f"  日志: {log_path}\n")

    results = {}   # (char, step) → True/False/None(skipped)
    aborted = False

    for step in steps:
        banner(tee, f"步骤: {step.upper()}", char="-")
        for char in chars:
            ok = run_step(tee, char, step)
            results[(char, step)] = ok
            if not ok and not args.skip_errors:
                tee.write(f"\n  [!] {char} {step.upper()} 失败，中止训练。\n"
                          f"  使用 --skip-errors 可忽略错误继续运行。\n")
                aborted = True
                break
        if aborted:
            break

    # 汇总
    banner(tee, "训练结果汇总")
    header = f"  {'角色':<12}" + "".join(f"  {s.upper():<6}" for s in steps)
    tee.write(header + "\n")
    tee.write("  " + "-" * (len(header) - 2) + "\n")
    for char in chars:
        row = f"  {char:<12}"
        for step in steps:
            r = results.get((char, step))
            if r is None:
                icon = "  ----"
            elif r:
                icon = "  ✓   "
            else:
                icon = "  ✗   "
            row += icon
        tee.write(row + "\n")

    total = len(results)
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    tee.write(f"\n  总计: {total} 步，{passed} 成功，{failed} 失败\n")
    tee.write(f"  日志已保存至: {log_path}\n")

    tee.close()
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
