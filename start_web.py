#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ST-Raptor Web应用启动脚本
使用HTML/CSS/JavaScript前端，不依赖Gradio
"""

import os
import sys

# 确保可以导入项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clean_cache import clear_cache_folder
from utils.constants import CACHE_DIR

# 启动前清理 cache，避免残留旧中间文件
clear_cache_folder(CACHE_DIR)

from web_app import app
import uvicorn

def main():
    # 注意：导入app时会初始化日志系统，所以不要在导入前清理日志
    # 如果需要清理旧日志，应该在日志系统初始化之后进行
    
    print("🚀 启动 ST-Raptor Web 界面...")
    print("📋 访问地址: http://localhost:7860")
    print("⏹️  按 Ctrl+C 停止服务")
    print("✨ 使用现代化的暗色主题HTML前端")
    
    # 导入app后，日志系统已经初始化，现在可以记录启动信息
    from loguru import logger
    logger.info("=" * 60)
    logger.info("ST-Raptor Web应用启动")
    logger.info("=" * 60)
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")
    except (KeyboardInterrupt, Exception) as e:
        if not isinstance(e, KeyboardInterrupt):
            print(f"服务运行出错: {e}")
    finally:
        print("\n🛑 正在关闭界面并清理环境...")
        print("✅ 服务已安全停止。")

if __name__ == "__main__":
    main()
