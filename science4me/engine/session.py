import datetime
import json
import os
import sys
from typing import Any, Dict, List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table

from science4me.engine.llm import LLMClient
from science4me.engine.state import ResearchState
from science4me.agents.ideation import IdeationAgent
from science4me.agents.planner import PlannerAgent
from science4me.agents.coder import CoderAgent
from science4me.agents.visualizer import VisualizerAgent
from science4me.tools.sandbox import SandboxExecutor
from science4me.tools.exporter import ResearchExporter

console = Console(highlight=False)


class ResearchSession:
    """
    人机协同 (Human-in-the-Loop) 交互式科研副驾驶会话控制器
    分阶段执行：每一步输出方案，交由研究者审查、微调或授权后推进
    """

    def __init__(self, topic: str, workspace_root: str = "workspaces", config_path: str = "configs/config.yaml"):
        self.topic = topic
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"run_{self.timestamp}"
        self.work_dir = os.path.abspath(os.path.join(workspace_root, self.run_id))
        os.makedirs(self.work_dir, exist_ok=True)

        self.llm = LLMClient(config_path)
        self.ideation_agent = IdeationAgent(self.llm)
        self.planner_agent = PlannerAgent(self.llm)
        self.coder_agent = CoderAgent(self.llm)
        self.visualizer_agent = VisualizerAgent()

        self.state: ResearchState = {
            "topic": topic,
            "run_id": self.run_id,
            "work_dir": self.work_dir,
            "status": "initialized",
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

    # ==================== 阶段 1：选题与假说梳理 ====================
    def stage_ideation(self) -> Dict[str, Any]:
        console.print(Panel(
            f"[bold cyan]正在为课题进行文献调研与假说发散...[/bold cyan]\n课题方向: [yellow]{self.topic}[/yellow]",
            title="[bold green]阶段 1 / 4：选题与文献调研[/bold green]"
        ))

        res = self.ideation_agent.run(self.topic, max_papers=5)
        self.state["papers"] = res["papers"]
        idea = res["idea"]
        self.state["idea"] = idea

        # 展示文献
        console.print(f"\n[bold]检索到 {len(res['papers'])} 篇相关前沿论文:[/bold]")
        for idx, p in enumerate(res["papers"][:3], 1):
            console.print(f"  {idx}. [cyan]{p.get('title')}[/cyan] ({p.get('published')})")

        # 呈现待审核假说
        rqs = "\n".join([f"    - {rq}" for rq in idea.get("research_questions", [])])
        console.print(Panel(
            f"[bold]拟定论文标题:[/bold] {idea.get('proposed_title')}\n\n"
            f"[bold]研究动机 (Motivation):[/bold]\n  {idea.get('motivation')}\n\n"
            f"[bold]核心假说与方法 (Proposed Method):[/bold]\n  {idea.get('proposed_method')}\n\n"
            f"[bold]核心研究问题 (RQs):[/bold]\n{rqs}",
            title="[bold yellow]★ 人工门禁 1：假说方案审查[/bold yellow]"
        ))

        choice = Prompt.ask(
            "\n请做出决策",
            choices=["确认通过", "修改假说", "重新发散"],
            default="确认通过"
        )

        if choice == "修改假说":
            user_title = Prompt.ask("请输入自定义的修改标题 (回车沿用原标题)", default=idea.get("proposed_title"))
            user_method = Prompt.ask("请输入对创新方法的调整说明 (回车沿用原方案)", default=idea.get("proposed_method"))
            self.state["idea"]["proposed_title"] = user_title
            self.state["idea"]["proposed_method"] = user_method
            console.print("[green]已更新您的自定义假说设置！[/green]")
        elif choice == "重新发散":
            new_prompt = Prompt.ask("请输入进一步细化的选题提示词", default=self.topic)
            self.topic = new_prompt
            return self.stage_ideation()

        return self.state["idea"]

    # ==================== 阶段 2：实验方案设计 ====================
    def stage_planning(self) -> Dict[str, Any]:
        console.print(Panel(
            "[bold cyan]正在根据确认的假说设计实验方案、对照基线与消融实验...[/bold cyan]",
            title="[bold green]阶段 2 / 4：实验方案规划[/bold green]"
        ))

        plan = self.planner_agent.run(self.state["idea"])
        self.state["experiment_plan"] = plan

        baselines_text = ", ".join([b.get("name") for b in plan.get("baselines", [])])
        ablations_text = ", ".join([a.get("name") for a in plan.get("ablations", [])])
        hyp = plan.get("hyperparameters", {})

        console.print(Panel(
            f"[bold]评测基准任务:[/bold] {plan.get('benchmark_task')}\n"
            f"[bold]数据集类型:[/bold] {plan.get('dataset_source')}\n"
            f"[bold]对比基线模型:[/bold] {baselines_text}\n"
            f"[bold]消融实验分支:[/bold] {ablations_text}\n"
            f"[bold]评价指标:[/bold] {', '.join(plan.get('evaluation_metrics', []))}\n"
            f"[bold]默认超参数:[/bold] Epochs={hyp.get('epochs')}, BatchSize={hyp.get('batch_size')}, LR={hyp.get('learning_rate')}",
            title="[bold yellow]★ 人工门禁 2：实验方案与参数审查[/bold yellow]"
        ))

        if Confirm.ask("是否需要调整超参数？", default=False):
            epochs = Prompt.ask("训练轮次 (Epochs)", default=str(hyp.get("epochs", 15)))
            lr = Prompt.ask("学习率 (Learning Rate)", default=str(hyp.get("learning_rate", 0.002)))
            self.state["experiment_plan"]["hyperparameters"]["epochs"] = int(epochs)
            self.state["experiment_plan"]["hyperparameters"]["learning_rate"] = float(lr)
            console.print("[green]超参数已更新！[/green]")

        return self.state["experiment_plan"]

    # ==================== 阶段 3：代码编写与沙箱验证 ====================
    def stage_coding_and_execution(self) -> bool:
        console.print(Panel(
            "[bold cyan]正在编写自包含 PyTorch 实验代码并设置数据流...[/bold cyan]",
            title="[bold green]阶段 3 / 4：代码编写与沙箱运行[/bold green]"
        ))

        code = self.coder_agent.generate_code(self.state["idea"], self.state["experiment_plan"])
        self.state["code_script"] = code

        script_path = os.path.join(self.work_dir, "experiment.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)

        console.print(f"[cyan]实验脚本已生成在:[/cyan] {script_path}")
        console.print(f"[dim]代码体积: {len(code.splitlines())} 行 | 涵盖所有 Baseline、Proposed Model 与 Ablation 分支[/dim]")

        # 人工审查门禁
        if not Confirm.ask("\n代码审查完成，是否授权启动本地沙箱运行该实验？", default=True):
            console.print("[yellow]已暂停执行。您可以手动编辑 experiment.py 后再继续推进。[/yellow]")
            return False

        console.print("\n[bold magenta]>> 正在沙箱中启动实验训练...[/bold magenta]")
        success, logs, metrics, error_tb = SandboxExecutor.run_script(
            code=code,
            work_dir=self.work_dir,
            timeout_seconds=300
        )

        self.state["execution_success"] = success
        self.state["execution_logs"] = logs
        self.state["metrics"] = metrics
        self.state["error_traceback"] = error_tb

        if success:
            console.print("[bold green][OK] 实验执行成功！所有模型均已完成收敛并输出指标数据。[/bold green]")
        else:
            console.print(f"[bold red][FAIL] 实验报错:[/bold red] {error_tb}")
            if Confirm.ask("是否启用 AI 自愈调试器自动修复代码并重跑？", default=True):
                fixed_code = self.coder_agent.fix_code(code, error_tb, self.state["experiment_plan"])
                self.state["code_script"] = fixed_code
                return self.stage_coding_and_execution()

        return success

    # ==================== 阶段 4：数据分析与决策复盘 ====================
    def stage_analysis_and_decision(self):
        console.print(Panel(
            "[bold cyan]正在解析训练轨迹、绘制学术图表并计算统计对比...[/bold cyan]",
            title="[bold green]阶段 4 / 4：数据分析与决策复盘[/bold green]"
        ))

        plots, tables = self.visualizer_agent.run(self.state["metrics"], self.work_dir)
        self.state["plots"] = plots
        self.state["latex_tables"] = tables

        results = self.state["metrics"].get("results", {})
        if results:
            table = Table(title="实验量化评测对比看板", show_header=True, header_style="bold cyan")
            table.add_column("模型名称", style="dim")
            table.add_column("测试准确率 (Accuracy)", justify="right")
            table.add_column("测试损失 (Loss)", justify="right")
            table.add_column("训练耗时 (s)", justify="right")

            for name, m in results.items():
                acc = f"{m.get('test_accuracy', 0)*100:.2f}%"
                loss = f"{m.get('test_loss', 0):.4f}"
                t = f"{m.get('training_time_sec', 0):.2f}"
                table.add_row(name, acc, loss, t)
            console.print("\n", table)

        console.print(f"\n[cyan]生成的科研矢量图表:[/cyan]")
        for p in plots:
            console.print(f"  - {os.path.join(self.work_dir, p)}")

        # 决策门禁
        choice = Prompt.ask(
            "\n★ 人工门禁 4：实验结论与后续决策",
            choices=["导出成果报告", "调整参数再测", "追加新消融实验"],
            default="导出成果报告"
        )

        if choice == "导出成果报告":
            ResearchExporter.export_all(
                work_dir=self.work_dir,
                paper_markdown=f"# {self.state['idea'].get('proposed_title')}\n\n## 实验结果\n{json.dumps(results, indent=2)}",
                paper_latex="",
                paper_bibtex="",
                metrics=self.state["metrics"],
                review={"decision": "Accepted by Researcher"},
                plots=plots
            )
            console.print(Panel(
                f"[bold green]实验闭环圆满完成！[/bold green]\n"
                f"产物工作区: {self.work_dir}\n"
                f"核心指标文件: {os.path.join(self.work_dir, 'metrics.json')}\n"
                f"成果图表目录: {os.path.join(self.work_dir, 'figures')}",
                title="[bold green]科研阶段总结[/bold green]"
            ))
        elif choice == "调整参数再测":
            self.stage_planning()
            self.stage_coding_and_execution()
            self.stage_analysis_and_decision()
