import asyncio
import datetime
import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Set
from fastapi import WebSocket

from science4me.engine.llm import LLMClient
from science4me.engine.evolution_log import EvolutionHistoryManager
from science4me.agents.goal_architecture import GoalArchitectureAgent
from science4me.agents.experiment_planner import ExperimentPlannerAgent
from science4me.agents.execution_training import ExecutionTrainingAgent
from science4me.agents.optimization_diagnosis import OptimizationDiagnosisAgent
from science4me.engine.workflow import ResearchWorkflow
from science4me.engine.state import ResearchState


class WebSessionRunner:
    """
    Web 端科研实验会话控制器
    支持人机协同演进（带三大人机门禁异步挂起与唤醒）与全自动科研流水线
    """

    def __init__(self, workspace_root: str = "workspaces", config_path: str = "configs/config.yaml"):
        self.workspace_root = workspace_root
        self.config_path = config_path

        # 运行状态
        self.status: str = "idle"  # idle, running, waiting_gate, completed, terminated, error
        self.mode: str = "interactive"  # interactive, autonomous
        self.run_id: str = ""
        self.work_dir: str = ""
        self.topic: str = ""
        self.current_stage: str = "idle"  # idle, stage_1, stage_2, stage_3, stage_4, completed
        self.generation: int = 1
        self.max_generations: int = 5

        # 核心产物数据
        self.goal_data: Dict[str, Any] = {}
        self.current_plan: Dict[str, Any] = {}
        self.current_architecture: Dict[str, Any] = {}
        self.changes_description: str = "初始代架构"
        self.latest_metrics: Dict[str, Any] = {}
        self.evolution_history: List[Dict[str, Any]] = []
        self.plots: List[str] = []
        self.paper_markdown: str = ""
        self.paper_latex: str = ""
        self.bundle_path: str = ""

        # 人机门禁状态
        self.gate_info: Optional[Dict[str, Any]] = None
        self._gate_event: Optional[asyncio.Event] = None
        self._gate_response: Optional[Dict[str, Any]] = None

        # 日志缓冲区与 WebSocket 连接管理
        self.logs: List[str] = []
        self.active_websockets: Set[WebSocket] = set()

        # 异步任务句柄
        self._task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def register_websocket(self, ws: WebSocket):
        self.active_websockets.add(ws)

    def unregister_websocket(self, ws: WebSocket):
        self.active_websockets.discard(ws)

    async def broadcast(self, event_type: str, payload: Any):
        """向所有连接的 Web 客户端广播事件"""
        msg = json.dumps({"type": event_type, "payload": payload}, ensure_ascii=False)
        stale = []
        for ws in self.active_websockets:
            try:
                await ws.send_text(msg)
            except Exception:
                stale.append(ws)
        for s in stale:
            self.active_websockets.discard(s)

    def log(self, text: str):
        """记录日志并向前端广播"""
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        formatted = f"[{ts}] {text}"
        self.logs.append(formatted)
        if len(self.logs) > 2000:
            self.logs = self.logs[-2000:]

        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self.broadcast("log", {"line": formatted}),
                self._loop
            )

    def get_state(self) -> Dict[str, Any]:
        """获取当前完整状态快照"""
        return {
            "status": self.status,
            "mode": self.mode,
            "run_id": self.run_id,
            "work_dir": self.work_dir,
            "topic": self.topic,
            "current_stage": self.current_stage,
            "generation": self.generation,
            "max_generations": self.max_generations,
            "goal_data": self.goal_data,
            "current_plan": self.current_plan,
            "current_architecture": self.current_architecture,
            "latest_metrics": self.latest_metrics,
            "evolution_history": self.evolution_history,
            "gate_info": self.gate_info,
            "plots": self.plots,
            "paper_markdown": self.paper_markdown,
            "bundle_path": self.bundle_path,
            "logs_count": len(self.logs)
        }

    async def start(
        self,
        topic: str,
        mode: str = "interactive",
        target_metric: str = "test_accuracy",
        target_threshold: float = 0.95,
        target_condition: str = ">=",
        epochs: int = 15,
        learning_rate: float = 0.002,
        batch_size: int = 64,
        max_generations: int = 5
    ) -> Dict[str, Any]:
        """启动新实验任务"""
        if self.status == "running" or self.status == "waiting_gate":
            return {"success": False, "message": "已有实验正在运行中"}

        self._loop = asyncio.get_running_loop()
        self.status = "running"
        self.mode = mode
        self.topic = topic
        self.generation = 1
        self.max_generations = max_generations
        self.logs = []
        self.gate_info = None
        self._gate_event = None
        self._gate_response = None
        self.latest_metrics = {}
        self.evolution_history = []
        self.plots = []
        self.paper_markdown = ""
        self.paper_latex = ""
        self.bundle_path = ""

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"run_{timestamp}"
        self.work_dir = os.path.abspath(os.path.join(self.workspace_root, self.run_id))
        os.makedirs(self.work_dir, exist_ok=True)

        self.custom_hypers = {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size
        }
        self.custom_target_goal = {
            "metric": target_metric,
            "threshold": target_threshold,
            "condition": target_condition,
            "description": f"{target_metric} {target_condition} {target_threshold*100:.1f}%"
        }

        self.log(f"实验启动: {self.topic}")
        self.log(f"工作区: {self.work_dir} | 模式: {self.mode}")
        await self.broadcast("status_change", self.get_state())

        if mode == "interactive":
            self._task = asyncio.create_task(self._run_interactive_pipeline())
        else:
            self._task = asyncio.create_task(self._run_autonomous_pipeline())

        return {"success": True, "run_id": self.run_id}

    async def submit_gate_decision(self, gate_id: int, decision: str, custom_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """接收前端提交的人机门禁决策并解除挂起"""
        if not self.gate_info or self.gate_info.get("gate_id") != gate_id:
            return {"success": False, "message": "当前未在等待该门禁的决策"}

        if not self._gate_event or self._gate_event.is_set():
            return {"success": False, "message": "门禁事件已失效"}

        self._gate_response = {
            "decision": decision,
            "custom_data": custom_data or {}
        }
        self.log(f"收到门禁 {gate_id} 决策: {decision}")
        self._gate_event.set()
        return {"success": True}

    async def stop(self) -> Dict[str, Any]:
        """终止当前实验"""
        if self.status not in ["running", "waiting_gate"]:
            return {"success": False, "message": "当前没有正在运行的实验"}

        self.status = "terminated"
        self.current_stage = "idle"
        self.log("用户已人工终止当前实验闭环。")

        if self._gate_event and not self._gate_event.is_set():
            self._gate_response = {"decision": "人工终止实验", "custom_data": {}}
            self._gate_event.set()

        if self._task and not self._task.done():
            self._task.cancel()

        await self.broadcast("status_change", self.get_state())
        return {"success": True}

    # ==================== 人机协同演进管线 ====================
    async def _run_interactive_pipeline(self):
        try:
            llm = LLMClient(self.config_path)
            evolution_mgr = EvolutionHistoryManager(self.work_dir)

            agent1_goal = GoalArchitectureAgent(llm)
            agent2_plan = ExperimentPlannerAgent(llm)
            agent3_train = ExecutionTrainingAgent(llm)
            agent4_diag = OptimizationDiagnosisAgent(llm)

            # ----- 阶段 1：选题与目标架构规划 -----
            while True:
                self.current_stage = "stage_1"
                self.status = "running"
                await self.broadcast("status_change", self.get_state())
                self.log("智能体 1 (GoalArchitectureAgent) 启动: 正在查阅前沿文献，确立课题与终止目标...")

                res = await asyncio.to_thread(agent1_goal.run, self.topic)
                self.goal_data = res

                # 若用户设置了自定义终止条件，应用之
                if self.custom_target_goal.get("threshold"):
                    self.goal_data["target_goal"] = {
                        "metric": self.custom_target_goal["metric"],
                        "threshold": self.custom_target_goal["threshold"],
                        "condition": self.custom_target_goal["condition"],
                        "description": f"{self.custom_target_goal['metric']} {self.custom_target_goal['condition']} {self.custom_target_goal['threshold']*100:.1f}%"
                    }

                self.current_architecture = res.get("initial_architecture", {})
                self.log(f"确立课题: {res.get('topic')}")
                self.log(f"停止目标: {self.goal_data.get('target_goal', {}).get('description')}")
                self.log(f"初始架构: {self.current_architecture.get('name')}")

                # 挂起进入门禁 1
                self.status = "waiting_gate"
                self.gate_info = {
                    "gate_id": 1,
                    "title": "人工审核门禁 1：课题、停止目标与架构审查",
                    "papers": res.get("papers", []),
                    "topic": res.get("topic"),
                    "target_goal": self.goal_data.get("target_goal", {}),
                    "initial_architecture": self.current_architecture,
                    "evolution_strategy": res.get("evolution_strategy", [])
                }
                self._gate_event = asyncio.Event()
                await self.broadcast("status_change", self.get_state())
                await self.broadcast("gate_required", self.gate_info)

                await self._gate_event.wait()
                resp = self._gate_response or {}
                dec = resp.get("decision")
                custom = resp.get("custom_data", {})

                if dec == "重新选题":
                    new_topic = custom.get("topic") or self.topic
                    self.topic = new_topic
                    self.log(f"研究者要求重新选题，新方向: {new_topic}")
                    continue
                elif dec == "修改目标与架构":
                    if custom.get("topic"):
                        self.goal_data["topic"] = custom.get("topic")
                    if custom.get("threshold"):
                        self.goal_data["target_goal"]["threshold"] = float(custom.get("threshold"))
                        self.goal_data["target_goal"]["description"] = f"{self.goal_data['target_goal']['metric']} {self.goal_data['target_goal']['condition']} {float(custom.get('threshold'))*100:.1f}%"
                    if custom.get("arch_name"):
                        self.current_architecture["name"] = custom.get("arch_name")
                    self.log("课题与量化停止目标已根据研究者意见更新。")

                self.gate_info = None
                break

            # ----- 阶段 2：实验方案规划 -----
            self.current_stage = "stage_2"
            self.status = "running"
            await self.broadcast("status_change", self.get_state())
            self.log("智能体 2 (ExperimentPlannerAgent) 启动: 正在制定实验方案与对比基线...")

            plan = await asyncio.to_thread(
                agent2_plan.run,
                topic=self.goal_data.get("topic", self.topic),
                target_goal=self.goal_data.get("target_goal", {}),
                architecture_spec=self.current_architecture
            )
            # 应用用户指定的超参数
            plan.setdefault("hyperparameters", {})
            for k, v in self.custom_hypers.items():
                if v is not None:
                    plan["hyperparameters"][k] = v

            self.current_plan = plan
            baselines = [b.get("name") for b in plan.get("baselines", [])]
            self.log(f"任务规划完成: 对比基准 {', '.join(baselines)}")

            # 挂起进入门禁 2
            self.status = "waiting_gate"
            self.gate_info = {
                "gate_id": 2,
                "title": "人工审核门禁 2：实验方案与硬件参数审查",
                "task_description": plan.get("task_description"),
                "baselines": plan.get("baselines", []),
                "architecture": self.current_architecture,
                "evaluation_metrics": plan.get("evaluation_metrics", []),
                "hyperparameters": plan.get("hyperparameters", {})
            }
            self._gate_event = asyncio.Event()
            await self.broadcast("status_change", self.get_state())
            await self.broadcast("gate_required", self.gate_info)

            await self._gate_event.wait()
            resp = self._gate_response or {}
            custom = resp.get("custom_data", {})
            if custom.get("epochs"):
                self.current_plan["hyperparameters"]["epochs"] = int(custom.get("epochs"))
            if custom.get("learning_rate"):
                self.current_plan["hyperparameters"]["learning_rate"] = float(custom.get("learning_rate"))
            self.log(f"实验参数核验通过: Epochs={self.current_plan['hyperparameters'].get('epochs')}, LR={self.current_plan['hyperparameters'].get('learning_rate')}")
            self.gate_info = None

            # ----- 阶段 3 与 4：演进优化闭环循环 -----
            target_goal = self.goal_data.get("target_goal", {})

            while self.generation <= self.max_generations:
                # 阶段 3: 代码训练
                self.current_stage = "stage_3"
                self.status = "running"
                await self.broadcast("status_change", self.get_state())
                self.log(f"智能体 3 (ExecutionTrainingAgent) 启动: 正在编写并执行第 {self.generation} 代实验训练...")
                self.log(f"当前架构: {self.current_architecture.get('name')} | 演进特征: {self.changes_description}")

                def stream_callback(line: str):
                    self.log(line)

                success, metrics, logs = await asyncio.to_thread(
                    agent3_train.generate_and_train,
                    generation=self.generation,
                    architecture_spec=self.current_architecture,
                    experiment_plan=self.current_plan,
                    work_dir=self.work_dir,
                    evolution_mgr=evolution_mgr,
                    target_goal=target_goal,
                    changes_description=self.changes_description,
                    log_callback=stream_callback
                )

                self.latest_metrics = metrics
                self.evolution_history = evolution_mgr.history
                await self.broadcast("metrics_update", {
                    "generation": self.generation,
                    "metrics": metrics,
                    "evolution_history": self.evolution_history
                })

                if not success:
                    self.log(f"第 {self.generation} 代训练脚本异常退出，准备进行诊断。")

                # 阶段 4: 模型缺陷诊断与演进优化
                self.current_stage = "stage_4"
                self.status = "running"
                await self.broadcast("status_change", self.get_state())
                self.log(f"智能体 4 (OptimizationDiagnosisAgent) 启动: 正在深度诊断第 {self.generation} 代架构缺陷...")

                diagnosis = await asyncio.to_thread(
                    agent4_diag.run,
                    current_generation=self.generation,
                    metrics=metrics,
                    evolution_mgr=evolution_mgr,
                    target_goal=target_goal,
                    topic=self.topic
                )

                # 检查是否达成最终目标
                if diagnosis.get("target_achieved"):
                    self.log(f"恭喜！模型已达成设定的最终量化目标！最终指标: {diagnosis.get('current_metric_value')*100:.2f}%")
                    self.status = "completed"
                    self.current_stage = "completed"
                    self._check_and_update_artifacts()
                    await self.broadcast("status_change", self.get_state())
                    return

                # 若未达标，挂起进入门禁 3
                self.status = "waiting_gate"
                next_arch = diagnosis.get("next_architecture", {})
                self.gate_info = {
                    "gate_id": 3,
                    "title": f"人工审核门禁 3：第 {self.generation} 代缺陷诊断与下一代方案审查",
                    "current_generation": self.generation,
                    "current_metric_value": diagnosis.get("current_metric_value", 0.0),
                    "anchor_baseline_gen": diagnosis.get("anchor_baseline_gen", self.generation),
                    "anchor_baseline_value": diagnosis.get("anchor_baseline_value", diagnosis.get("current_metric_value", 0.0)),
                    "is_regression": diagnosis.get("is_regression", False),
                    "last_resort_activated": diagnosis.get("last_resort_activated", False),
                    "target_goal": target_goal,
                    "is_stagnant": diagnosis.get("is_stagnant", False),
                    "model_flaws": diagnosis.get("model_flaws", []),
                    "bottleneck_analysis": diagnosis.get("bottleneck_analysis", ""),
                    "next_architecture": next_arch
                }
                self._gate_event = asyncio.Event()
                await self.broadcast("status_change", self.get_state())
                await self.broadcast("gate_required", self.gate_info)

                await self._gate_event.wait()
                resp = self._gate_response or {}
                dec = resp.get("decision")
                custom = resp.get("custom_data", {})

                if dec == "终止实验" or dec == "人工终止实验":
                    self.status = "terminated"
                    self.current_stage = "idle"
                    self.gate_info = None
                    self.log("研究者已决定终止演进闭环。")
                    self._check_and_update_artifacts()
                    await self.broadcast("status_change", self.get_state())
                    return
                elif dec == "调整改进方案":
                    self.current_architecture = {
                        "name": custom.get("name", next_arch.get("name")),
                        "core_mechanisms": custom.get("core_mechanisms", next_arch.get("core_mechanisms")),
                        "design_rationale": "人工指导微调方案",
                        "changes_from_previous": "人工指定架构变动"
                    }
                    self.changes_description = "人工指导微调方案"
                    self.generation += 1
                else:  # 批准演进并训练下一代
                    self.changes_description = next_arch.get("changes_from_previous", f"由 Gen{self.generation} 演进")
                    self.current_architecture = next_arch
                    self.generation += 1

                self.gate_info = None

            # 达到代数上限
            self.status = "completed"
            self.current_stage = "completed"
            self.log(f"实验达到设定代数上限 ({self.max_generations} 代)，实验圆满结束。")
            self._check_and_update_artifacts()
            await self.broadcast("status_change", self.get_state())

        except asyncio.CancelledError:
            self.status = "terminated"
            self.current_stage = "idle"
            self.log("实验任务已取消。")
            await self.broadcast("status_change", self.get_state())
        except Exception as e:
            self.status = "error"
            self.current_stage = "idle"
            self.log(f"实验运行异常: {str(e)}")
            await self.broadcast("status_change", self.get_state())

    # ==================== 全自动 LangGraph 管线 ====================
    async def _run_autonomous_pipeline(self):
        try:
            self.log("启动全自动 LangGraph 多智能体科研流转...")
            workflow = ResearchWorkflow(config_path=self.config_path)

            initial_state: ResearchState = {
                "topic": self.topic,
                "run_id": self.run_id,
                "work_dir": self.work_dir,
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

            def run_stream():
                updates = []
                for event in workflow.graph.stream(initial_state):
                    updates.append(event)
                return updates

            # 执行流转
            for event in await asyncio.to_thread(run_stream):
                for node_name, state_update in event.items():
                    self.log(f"已完成节点: {node_name}")
                    if node_name == "ideation":
                        self.current_stage = "stage_1"
                    elif node_name == "planner":
                        self.current_stage = "stage_2"
                    elif node_name in ["coder", "executor"]:
                        self.current_stage = "stage_3"
                    elif node_name in ["visualizer", "writer", "reviewer", "export"]:
                        self.current_stage = "stage_4"

                    if "metrics" in state_update and state_update["metrics"]:
                        self.latest_metrics = state_update["metrics"]
                    if "plots" in state_update:
                        self.plots = state_update["plots"]
                    if "paper_markdown" in state_update:
                        self.paper_markdown = state_update["paper_markdown"]

                    await self.broadcast("status_change", self.get_state())

            self.status = "completed"
            self.current_stage = "completed"
            self._check_and_update_artifacts()
            self.log("全自动科研流水线已全部执行完成！")
            await self.broadcast("status_change", self.get_state())

        except Exception as e:
            self.status = "error"
            self.current_stage = "idle"
            self.log(f"全自动流程异常: {str(e)}")
            await self.broadcast("status_change", self.get_state())

    def _check_and_update_artifacts(self):
        """扫描并记录工作区内的关键产物"""
        if not os.path.exists(self.work_dir):
            return

        fig_dir = os.path.join(self.work_dir, "figures")
        if os.path.exists(fig_dir):
            self.plots = [f for f in os.listdir(fig_dir) if f.endswith((".png", ".jpg", ".svg"))]

        paper_path = os.path.join(self.work_dir, "paper.md")
        if os.path.exists(paper_path):
            try:
                with open(paper_path, "r", encoding="utf-8") as f:
                    self.paper_markdown = f.read()
            except Exception:
                pass

        bundle_path = os.path.join(self.work_dir, "research_bundle.zip")
        if os.path.exists(bundle_path):
            self.bundle_path = bundle_path
