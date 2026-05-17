"""
launcher.py — PyInstaller 打包入口
打包后双击 EXE 自动启动浏览器，运行 Streamlit 应用
"""
import sys
import os
import threading
import webbrowser
import time


def resource_path(rel):
    """兼容 PyInstaller _MEIPASS 路径"""
    base = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base, rel)


def open_browser():
    time.sleep(3)
    webbrowser.open("http://localhost:8501")


if __name__ == "__main__":
    threading.Thread(target=open_browser, daemon=True).start()

    # 启动 streamlit
    from streamlit.web import cli as stcli
    app_path = resource_path("app.py")
    sys.argv = [
        "streamlit", "run", app_path,
        "--server.headless", "true",
        "--server.port", "8501",
        "--server.enableCORS", "false",
        "--browser.gatherUsageStats", "false",
    ]
    sys.exit(stcli.main())
