import sys
import os
import socket
import webbrowser
import threading


def find_free_port(default=8501):
    """尝试使用默认端口，被占用则自动找空闲端口"""
    for port in range(default, default + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    # fallback: OS 分配
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def open_browser(port):
    webbrowser.open(f"http://localhost:{port}")


def main():
    # ✅ 关键：frozen EXE 里用 sys._MEIPASS 定位资源
    if getattr(sys, "frozen", False):
        bundle_dir = sys._MEIPASS
    else:
        bundle_dir = os.path.dirname(os.path.abspath(__file__))

    app_path = os.path.join(bundle_dir, "app.py")
    port = find_free_port(8501)

    # 延迟 2 秒后自动打开浏览器
    threading.Timer(2.0, open_browser, args=[port]).start()

    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit",
        "run",
        app_path,
        f"--server.port={port}",
        "--server.headless=true",
        "--global.developmentMode=false",
        "--browser.gatherUsageStats=false",
    ]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
