"""
launcher.py — PyInstaller 打包入口
双击 EXE 自动启动浏览器并运行 Streamlit 应用
"""
import sys
import os
import threading
import webbrowser
import time

# ── 必须在任何 matplotlib import 之前设置后端 ──────────────────
# PyInstaller 环境下没有 display，必须强制 Agg
os.environ.setdefault("MPLBACKEND", "Agg")


def resource_path(rel):
    """兼容 PyInstaller _MEIPASS 与开发环境路径"""
    base = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base, rel)


def open_browser():
    time.sleep(4)          # 等 streamlit 启动完成
    webbrowser.open("http://localhost:8501")


if __name__ == "__main__":
    threading.Thread(target=open_browser, daemon=True).start()

    from streamlit.web import cli as stcli
    app_path = resource_path("app.py")
    sys.argv = [
        "streamlit", "run", app_path,
        "--server.headless",        "true",
        "--server.port",            "8501",
        "--server.enableCORS",      "false",
        "--server.enableXsrfProtection", "false",
        "--browser.gatherUsageStats", "false",
    ]
    sys.exit(stcli.main())
