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
from science4me.engine.evolution_log import EvolutionHistoryManager
from science4me.agents.goal_architecture import GoalArchitectureAgent
from science4me.agents.experiment_planner import ExperimentPlannerAgent
from science4me.agents.execution_training import ExecutionTrainingAgent
from science4me.agents.optimization_diagnosis import OptimizationDiagnosisAgent

console = Console(highlight=False)


class ResearchSession:
    """
    第一阶段核心：架构自主演进与实验训练人机协同控制器
    驱动 4 大智能体协作闭环：
    Agent 1 (目标架构) -> Agent 2 (方案规划) -> Loop [ Agent 3 (训练执行) -> Agent 4 (诊断优化与停滞突围) ]
    """

    def __init__(self, topic: str, workspace_root: str = "workspaces", config_path: str = "configs/config.yaml"):
        self.user_topic = topic
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"run_{self.timestamp}"
        self.work_dir = os.path.abspath(os.path.join(workspace_root, self.run_id))
        os.makedirs(self.work_dir, exist_ok=True)

        self.llm = LLMClient(config_path)
        self.evolution_mgr = EvolutionHistoryManager(self.work_dir)

        # 初始化第一阶段四大智能体
        self.agent1_goal = GoalArchitectureAgent(self.llm)
        self.agent2_plan = ExperimentPlannerAgent(self.llm)
        self.agent3_train = ExecutionTrainingAgent(self.llm)
        self.agent4_diag = OptimizationDiagnosisAgent(self.llm)

        # 会话运行状态
        self.generation = 1
        self.goal_data: Dict[str, Any] = {}
        self.current_plan: Dict[str, Any] = {}
        self.current_architecture: Dict[str, Any] = {}
        self.changes_description: str = "初始代基础架构"

    # ==================== 智能体 1：选题、目标设定与架构规划 ====================
    def stage_goal_and_architecture(self) -> Dict[str, Any]:
        console.print(Panel(
            f"[bold cyan]正在查阅前沿文献，确立科研课题、停止目标与架构演进路线...[/bold cyan]\n"
            f"输入方向: [yellow]{self.user_topic}[/yellow]",
            title="[bold green]智能体 1：选题与目标架构规划 (GoalArchitectureAgent)[/bold green]"
        ))

        res = self.agent1_goal.run(self.user_topic)
        self.goal_data = res
        self.current_architecture = res.get("initial_architecture", {})

        papers = res.get("papers", [])
        target = res.get("target_goal", {})
        evo = res.get("evolution_strategy", [])

        # 呈现检索文献
        console.print(f"\n[bold]检索到 {len(papers)} 篇相关前沿论文:[/bold]")
        for idx, p in enumerate(papers[:3], 1):
            console.print(f"  {idx}. [cyan]{p.get('title')}[/cyan] ({p.get('published')})")

        # 呈现演进路线
        evo_lines = "\n".join([f"    - 代际 {item.get('generation')}: [cyan]{item.get('architecture')}[/cyan] ({item.get('focus')})" for item in evo])

        console.print(Panel(
            f"[bold]确立课题:[/bold] {res.get('topic')}\n\n"
            f"[bold red]★ 最终量化停止目标:[/bold red] {target.get('description')} "
            f"({target.get('metric')} {target.get('condition')} {target.get('threshold')})\n\n"
            f"[bold]初始代架构 (Gen 1):[/bold] {self.current_architecture.get('name')}\n"
            f"  - 核心机制: {self.current_architecture.get('core_mechanisms')}\n"
            f"  - 设计动机: {self.current_architecture.get('design_rationale')}\n\n"
            f"[bold]预期架构演进路线:[/bold]\n{evo_lines}",
            title="[bold yellow]★ 人工审核门禁 1：课题、停止目标与架构审查[/bold yellow]"
        ))

        choice = Prompt.ask(
            "\n请做出审核决策",
            choices=["确认通过", "修改目标与架构", "重新发散"],
            default="确认通过"
        )

        if choice == "修改目标与架构":
            new_title = Prompt.ask("自定义课题名称 (回车沿用)", default=res.get("topic"))
            new_target = Prompt.ask("自定义停止准确率阈值 (0.0~1.0)", default=str(target.get("threshold", 0.95)))
            self.goal_data["topic"] = new_title
            self.goal_data["target_goal"]["threshold"] = float(new_target)
            console.print("[green]课题与最终停止目标已更新！[/green]")
        elif choice == "重新发散":
            self.user_topic = Prompt.ask("输入新的探索方向关键词", default=self.user_topic)
            return self.stage_goal_and_architecture()

        return self.goal_data

    # ==================== 智能体 2：实验方案设计 ====================
    def stage_experiment_planning(self) -> Dict[str, Any]:
        console.print(Panel(
            "[bold cyan]正在结合文献、选题与目标制定具体实验评测方案...[/bold cyan]",
            title="[bold green]智能体 2：实验方案规划 (ExperimentPlannerAgent)[/bold green]"
        ))

        plan = self.agent2_plan.run(
            topic=self.goal_data.get("topic", self.user_topic),
            target_goal=self.goal_data.get("target_goal", {}),
            architecture_spec=self.current_architecture
        )
        self.current_plan = plan

        baselines_text = ", ".join([b.get("name") for b in plan.get("baselines", [])])
        hyp = plan.get("hyperparameters", {})

        console.print(Panel(
            f"[bold]任务描述:[/bold] {plan.get('task_description')}\n"
            f"[bold]对比基准模型:[/bold] {baselines_text}\n"
            f"[bold]当前代待测架构:[/bold] {self.current_architecture.get('name')}\n"
            f"[bold]评价指标:[/bold] {', '.join(plan.get('evaluation_metrics', []))}\n"
            f"[bold]硬件偏好:[/bold] {hyp.get('device_preference')} (自动检测 CUDA)\n"
            f"[bold]默认超参数:[/bold] Epochs={hyp.get('epochs')}, BatchSize={hyp.get('batch_size')}, LR={hyp.get('learning_rate')}",
            title="[bold yellow]★ 人工审核门禁 2：实验方案与硬件参数审查[/bold yellow]"
        ))

        if Confirm.ask("是否需要调整训练超参数？", default=False):
            epochs = Prompt.ask("训练轮次 (Epochs)", default=str(hyp.get("epochs", 15)))
            lr = Prompt.ask("学习率 (Learning Rate)", default=str(hyp.get("learning_rate", 0.002)))
            self.current_plan["hyperparameters"]["epochs"] = int(epochs)
            self.current_plan["hyperparameters"]["learning_rate"] = float(lr)
            console.print("[green]超参数已调整完毕！[/green]")

        return self.current_plan

    # ==================== 智能体 3 与 4：演进优化循环 ====================
    def run_evolution_loop(self, max_generations: int = 5):
        """执行 Agent 3 (训练与持久化) 与 Agent 4 (诊断与演进) 闭环迭代"""
        target_goal = self.goal_data.get("target_goal", {})

        while self.generation <= max_generations:
            console.print(Panel(
                f"[bold magenta]>> 启动第 {self.generation} 代架构实验与模型训练...[/bold magenta]\n"
                f"当前架构: [cyan]{self.current_architecture.get('name')}[/cyan]\n"
                f"架构说明: {self.current_architecture.get('core_mechanisms')}\n"
                f"演进变更: {self.changes_description}",
                title=f"[bold green]智能体 3：代码执行与训练 (Gen {self.generation})[/bold green]"
            ))

            # 1. 智能体 3：生成代码、沙箱执行并持久化记录
            success, metrics, logs = self.agent3_train.generate_and_train(
                generation=self.generation,
                architecture_spec=self.current_architecture,
                experiment_plan=self.current_plan,
                work_dir=self.work_dir,
                evolution_mgr=self.evolution_mgr,
                target_goal=target_goal,
                changes_description=self.changes_description
            )

            if not success:
                console.print(f"[bold red]第 {self.generation} 代训练脚本执行异常，请检查代码或环境。[/bold red]")
                if not Confirm.ask("是否继续重试？", default=True):
                    break

            # 呈现本代指标对比
            results = metrics.get("results", {})
            if results:
                table = Table(title=f"第 {self.generation} 代模型量化评测结果", show_header=True, header_style="bold cyan")
                table.add_column("模型名称", style="dim")
                table.add_column("测试准确率 (Accuracy)", justify="right")
                table.add_column("测试损失 (Loss)", justify="right")
                table.add_column("训练耗时 (s)", justify="right")
                for name, m in results.items():
                    acc = f"{m.get('test_accuracy', 0)*100:.2f}%"
                    loss = f"{m.get('test_loss', 0):.4f}"
                    t = f"{m.get('training_time_sec', 0):.2f}"
                    table.add_row(name, acc, loss, t)
                console.print(table)

            console.print(f"[dim]演进历史已持久化至: {self.evolution_mgr.log_file}[/dim]\n")

            # 2. 智能体 4：模型缺陷诊断与演进优化
            console.print(Panel(
                f"[bold cyan]正在诊断第 {self.generation} 代模型缺陷并评估演进方案...[/bold cyan]",
                title=f"[bold green]智能体 4：模型诊断与演进优化 (OptimizationDiagnosisAgent)[/bold green]"
            ))

            diagnosis = self.agent4_diag.run(
                current_generation=self.generation,
                metrics=metrics,
                evolution_mgr=self.evolution_mgr,
                target_goal=target_goal
            )

            # 检查是否达成最终目标 (停止条件)
            if diagnosis.get("target_achieved"):
                console.print(Panel(
                    f"[bold green]恭喜！模型已达成设定的最终量化目标！[/bold green]\n\n"
                    f"最终指标: [bold cyan]{target_goal.get('metric')} = {diagnosis.get('current_metric_value')*100:.2f}%[/bold cyan] "
                    f"(目标要求: {target_goal.get('condition')} {target_goal.get('threshold')*100:.2f}%)\n"
                    f"优胜架构: [yellow]{self.current_architecture.get('name')}[/yellow]\n"
                    f"演进代数: 共历经 {self.generation} 代演进\n"
                    f"全量演进历史: {self.evolution_mgr.log_file}",
                    title="[bold green]★ 科研目标顺利达成 (Stopping Target Achieved)[/bold green]"
                ))
                break

            # 若未达标，呈现缺陷分析与下一步方案
            flaws_text = "\n".join([f"    - {f}" for f in diagnosis.get("model_flaws", [])])
            is_stagnant = diagnosis.get("is_stagnant", False)
            next_arch = diagnosis.get("next_architecture", {})

            stagnant_warning = ""
            if is_stagnant:
                stagnant_warning = "\n[bold red]【停滞突围机制触发】：检测到近期实验指标未有明显提升，系统已生成跳脱局部的重构方案！[/bold red]\n"

            console.print(Panel(
                f"当前指标: {target_goal.get('metric')} = {diagnosis.get('current_metric_value')*100:.2f}% "
                f"(距离最终目标 {target_goal.get('threshold')*100:.2f}% 尚有差距)\n"
                f"{stagnant_warning}\n"
                f"[bold]诊断出的模型缺陷:[/bold]\n{flaws_text}\n\n"
                f"[bold]瓶颈深度分析:[/bold]\n  {diagnosis.get('bottleneck_analysis')}\n\n"
                f"[bold]拟建议的第 {self.generation + 1} 代演进架构:[/bold] [yellow]{next_arch.get('name')}[/yellow]\n"
                f"  - 核心机制: {next_arch.get('core_mechanisms')}\n"
                f"  - 设计依据: {next_arch.get('design_rationale')}\n"
                f"  - 相较上一代变更: {next_arch.get('changes_from_previous')}",
                title=f"[bold yellow]★ 人工审核门禁 3：第 {self.generation} 代缺陷诊断与下一代方案审查[/bold yellow]"
            ))

            choice = Prompt.ask(
                "\n请做出审核决策",
                choices=["批准演进并训练下一代", "调整改进方案", "人工终止实验"],
                default="批准演进并训练下一代"
            )

            if choice == "批准演进并训练下一代":
                self.changes_description = next_arch.get("changes_from_previous", f"由 Gen{self.generation} 演进")
                self.current_architecture = next_arch
                self.generation += 1
            elif choice == "调整改进方案":
                custom_name = Prompt.ask("输入自定义架构名称", default=next_arch.get("name"))
                custom_mech = Prompt.ask("输入自定义核心机制", default=next_arch.get("core_mechanisms"))
                self.current_architecture = {
                    "name": custom_name,
                    "core_mechanisms": custom_mech,
                    "design_rationale": "人工指导微调方案",
                    "changes_from_previous": "人工指定架构变动"
                }
                self.changes_description = "人工指导微调方案"
                self.generation += 1
            else:
                console.print("[yellow]已由研究者人工终止实验闭环。[/yellow]")
                break
