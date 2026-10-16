import argparse
import datetime
import io
import os
import sys

# Windows 终端 GBK 编码防爆处理
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from science4me.engine.workflow import ResearchWorkflow
from science4me.engine.state import ResearchState

console = Console(highlight=False)


def print_banner():
    banner = r"""
   _____ ______  _____ ______ _   _  _____ ______ _  _   __  __ ______ 
  / ____/ _____||_   _|  ____| \ | |/ ____|  ____| || | |  \/  |  ____|
 | (___| |        | | | |__  |  \| | |    | |__  | || |_| \  / | |__   
  \___ \ |        | | |  __| | . ` | |    |  __| |__   _| |\/| |  __|  
  ____) | |____  _| |_| |____| |\  | |____| |____   | | | |  | | |____ 
 |_____/ \_____||_____|______|_| \_|\_____|______|  |_| |_|  |_|______|
       === Autonomous AI Scientist & Research Engine (v0.1.0) ===
    """
    console.print(f"[bold cyan]{banner}[/bold cyan]")


def parse_args():
    parser = argparse.ArgumentParser(description="Science4Me 全自动科研与论文生成引擎")
    parser.add_argument(
        "--topic", "-t",
        type=str,
        default="Adaptive Learning Rate and Feature Variance Modulation in Deep Neural Networks",
        help="初始科研选题或核心假说"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--workspace", "-w",
        type=str,
        default="workspaces",
        help="实验产物工作区根目录"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="启用人机协同 (Human-in-the-Loop) 交互模式，每一步由人工审查确认"
    )
    return parser.parse_args()


def main():
    print_banner()
    args = parse_args()

    # 0. 若启用人机协同交互模式
    if args.interactive:
        from science4me.engine.session import ResearchSession
        session = ResearchSession(
            topic=args.topic,
            workspace_root=args.workspace,
            config_path=args.config
        )
        session.stage_goal_and_architecture()
        session.stage_experiment_planning()
        session.run_evolution_loop()
        return

    # 1. 初始化实验运行目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"
    work_dir = os.path.abspath(os.path.join(args.workspace, run_id))
    os.makedirs(work_dir, exist_ok=True)

    console.print(Panel(
        f"[bold green]科研选题主题:[/bold green] {args.topic}\n"
        f"[bold blue]实验产物目录:[/bold blue] {work_dir}\n"
        f"[bold yellow]运行唯一标识:[/bold yellow] {run_id}",
        title="[bold]实验初始化 (Research Initialization)[/bold]",
        expand=False
    ))

    # 2. 构建工作流
    try:
        workflow = ResearchWorkflow(config_path=args.config)
    except Exception as e:
        console.print(f"[bold red]工作流引擎初始化失败:[/bold red] {e}")
        sys.exit(1)

    initial_state: ResearchState = {
        "topic": args.topic,
        "run_id": run_id,
        "work_dir": work_dir,
        "status": "started",
        "papers": [],
        "idea": {},
        "experiment_plan": {},
        "code_script": "",
        "execution_logs": "",
        "execution_success": False,
        "error_traceback": "",
        "retry_count": 0,
        "metrics": {},
        "plots": [],
        "latex_tables": [],
        "paper_sections": {},
        "paper_bibtex": "",
        "paper_markdown": "",
        "paper_latex": "",
        "review": {},
        "revision_count": 0
    }

    # 3. 驱动 LangGraph 状态图流转
    console.print("\n[bold magenta]>> 正在启动多智能体科研全自动流转...[/bold magenta]\n")

    current_state = initial_state

    # 使用流式执行，打印各节点清晰阶段日志
    for event in workflow.graph.stream(initial_state):
        for node_name, state_update in event.items():
            current_state.update(state_update)
            console.print(f"[bold green][NODE][/bold green] 完成节点: [yellow]{node_name}[/yellow]")

            # 打印各节点关键输出
            if node_name == "ideation":
                idea = current_state.get("idea", {})
                papers = current_state.get("papers", [])
                console.print(f"  [cyan][OK] 选题成型:[/cyan] {idea.get('proposed_title')}")
                console.print(f"  [cyan][OK] 检索前沿文献:[/cyan] 成功捕获 {len(papers)} 篇 ArXiv 关联论文")
            elif node_name == "planner":
                plan = current_state.get("experiment_plan", {})
                baselines = [b.get("name") for b in plan.get("baselines", [])]
                console.print(f"  [cyan][OK] 实验规划:[/cyan] 基准对照组: {', '.join(baselines)}")
            elif node_name == "coder":
                console.print(f"  [cyan][OK] 代码生成:[/cyan] 独立实验脚本 experiment.py 已就绪 (重试轮次: {current_state.get('retry_count')})")
            elif node_name == "executor":
                succ = current_state.get("execution_success")
                status_text = "[green]执行成功[/green]" if succ else "[red]执行报错(触发自愈)[/red]"
                console.print(f"  [cyan][OK] 沙箱运行:[/cyan] 状态: {status_text}")
            elif node_name == "visualizer":
                plots = current_state.get("plots", [])
                console.print(f"  [cyan][OK] 学术制图:[/cyan] 已生成 {len(plots)} 张科研级图表")
            elif node_name == "writer":
                console.print(f"  [cyan][OK] 论文撰写:[/cyan] LaTeX 与 Markdown 双版本论文撰写排版完成")
            elif node_name == "reviewer":
                rev = current_state.get("review", {})
                score = rev.get("overall_score")
                decision = rev.get("decision")
                color = "green" if decision == "Accept" else "yellow"
                console.print(f"  [cyan][OK] 模拟评审:[/cyan] 综合得分: [{color}]{score}/10[/{color}] | 评审决议: [{color}]{decision}[/{color}]")
            elif node_name == "export":
                console.print(f"  [green][OK] 产物打包:[/green] research_bundle.zip 打包完成")

    # 4. 汇总展示研究成果
    metrics = current_state.get("metrics", {}).get("results", {})
    if metrics:
        table = Table(title="实验基线与创新模型量化评测对比", show_header=True, header_style="bold cyan")
        table.add_column("模型名称", style="dim")
        table.add_column("测试准确率 (Accuracy)", justify="right")
        table.add_column("测试损失 (Loss)", justify="right")
        table.add_column("训练耗时 (s)", justify="right")

        for name, m in metrics.items():
            acc = f"{m.get('test_accuracy', 0)*100:.2f}%"
            loss = f"{m.get('test_loss', 0):.4f}"
            t = f"{m.get('training_time_sec', 0):.2f}"
            table.add_row(name, acc, loss, t)
        console.print("\n", table)

    rev = current_state.get("review", {})
    console.print(Panel(
        f"[bold]论文标题:[/bold] {current_state.get('idea', {}).get('proposed_title')}\n"
        f"[bold]模拟评审决议:[/bold] [bold green]{rev.get('decision')}[/bold green] (总体评分: [bold green]{rev.get('overall_score')}/10[/bold green])\n"
        f"[bold]评审概述:[/bold] {rev.get('summary')}\n\n"
        f"[bold cyan]核心产物清单:[/bold cyan]\n"
        f"  - 完整 Markdown 论文: {os.path.join(work_dir, 'paper.md')}\n"
        f"  - 标准 LaTeX 源码:   {os.path.join(work_dir, 'paper.tex')}\n"
        f"  - BibTeX 文献库:     {os.path.join(work_dir, 'references.bib')}\n"
        f"  - 结构化实验指标:   {os.path.join(work_dir, 'metrics.json')}\n"
        f"  - 一键投递压缩包:   {os.path.join(work_dir, 'research_bundle.zip')}\n"
        f"  - 出版级图表目录:   {os.path.join(work_dir, 'figures')}",
        title="[bold green]科研任务圆满完成 (Research Mission Accomplished)[/bold green]"
    ))


if __name__ == "__main__":
    main()
