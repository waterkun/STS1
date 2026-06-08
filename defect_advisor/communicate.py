#!/usr/bin/env python3
"""
机器人（Defect）CommunicationMod 入口

委托给根目录的统一 communicate.py，自动识别 DEFECT 角色。
"""

import os
import sys

# 确保项目根目录在 sys.path 中
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# 直接复用根目录统一入口
import communicate

if __name__ == "__main__":
    communicate.main()
