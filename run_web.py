import argparse
import sys
import threading
import time
import webbrowser
import uvicorn
from rich.console import Console
from rich.panel import Panel

console = Console(highlight=False)


def open_browser(url: str, delay: float = 1.0):
    time.sleep(delay)
    try:
        webbrowser.open(url)
    except Exception:
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="Science4Me Web UI 服务启动脚本")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="绑定监听地址")
    parser.add_argument("--port", type=int, default=8000, help="服务端口")
    parser.add_argument("--no-browser", action="store_true", help="启动时不自动打开系统默认浏览器")
    return parser.parse_args()


def main():
    args = parse_args()
    url = f"http://{args.host}:{args.port}"

    banner = r"""
   _____ ______  _____ ______ _   _  _____ ______ _  _   __  __ ______ 
  / ____/ _____||_   _|  ____| \ | |/ ____|  ____| || | |  \/  |  ____|
 | (___| |        | | | |__  |  \| | |    | |__  | || |_| \  / | |__   
  \___ \ |        | | |  __| | . ` | |    |  __| |__   _| |\/| |  __|  
  ____) | |____  _| |_| |____| |\  | |____| |____   | | | |  | | |____ 
 |_____/ \_____||_____|______|_| \_|\_____|______|  |_| |_|  |_|______|
           === Web Dashboard & Interactive Engine ===
    """
    console.print(f"[bold cyan]{banner}[/bold cyan]")
    console.print(Panel(
        f"[bold green]Science4Me Web 看板服务正在启动...[/bold green]\n\n"
        f"本地访问链接: [bold cyan underline]{url}[/bold cyan underline]\n"
        f"API 文档端点: [cyan]{url}/docs[/cyan]\n"
        f"交互运行模式: 支持人机门禁协同演进 & 全自动科研流转",
        title="[bold]服务启动信息[/bold]",
        expand=False
    ))

    if not args.no_browser:
        threading.Thread(target=open_browser, args=(url,), daemon=True).start()

    uvicorn.run("science4me.web.server:app", host=args.host, port=args.port, reload=False, log_level="info")


if __name__ == "__main__":
    main()
