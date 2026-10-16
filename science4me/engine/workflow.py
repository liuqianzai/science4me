import os
import yaml
from typing import Any, Dict
from langgraph.graph import StateGraph, END

from science4me.engine.state import ResearchState
from science4me.engine.llm import LLMClient
from science4me.agents.ideation import IdeationAgent
from science4me.agents.planner import PlannerAgent
from science4me.agents.coder import CoderAgent
from science4me.agents.visualizer import VisualizerAgent
from science4me.agents.writer import WriterAgent
from science4me.agents.reviewer import ReviewerAgent
from science4me.tools.sandbox import SandboxExecutor
from science4me.tools.exporter import ResearchExporter


class ResearchWorkflow:
    """基于 LangGraph 编排的端到端全自动科研工作流状态机"""

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = self._load_config(config_path)
        self.llm = LLMClient(config_path)

        res_cfg = self.config.get("research", {})
        self.max_retries = int(res_cfg.get("max_retries", 3))
        self.timeout_seconds = int(res_cfg.get("timeout_seconds", 300))
        self.max_papers_search = int(res_cfg.get("max_papers_search", 5))
        self.acceptance_threshold = float(res_cfg.get("acceptance_threshold", 7.0))
        self.max_revisions = int(res_cfg.get("max_revisions", 2))

        # 初始化各环节智能体
        self.ideation_agent = IdeationAgent(self.llm)
        self.planner_agent = PlannerAgent(self.llm)
        self.coder_agent = CoderAgent(self.llm)
        self.visualizer_agent = VisualizerAgent()
        self.writer_agent = WriterAgent(self.llm)
        self.reviewer_agent = ReviewerAgent(self.llm)

        # 编译 LangGraph 计算图
        self.graph = self._build_graph()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def _node_ideation(self, state: ResearchState) -> Dict[str, Any]:
        """节点 1: 选题与文献检索"""
        topic = state["topic"]
        res = self.ideation_agent.run(topic, max_papers=self.max_papers_search)
        return {
            "papers": res["papers"],
            "idea": res["idea"],
            "status": "planning"
        }

    def _node_planner(self, state: ResearchState) -> Dict[str, Any]:
        """节点 2: 实验规划与基线制定"""
        plan = self.planner_agent.run(state["idea"])
        return {
            "experiment_plan": plan,
            "status": "coding"
        }

    def _node_coder(self, state: ResearchState) -> Dict[str, Any]:
        """节点 3: 代码生成与自愈调试"""
        retry_count = state.get("retry_count", 0)
        if retry_count == 0 or not state.get("error_traceback"):
            code = self.coder_agent.generate_code(state["idea"], state["experiment_plan"])
        else:
            code = self.coder_agent.fix_code(state["code_script"], state["error_traceback"], state["experiment_plan"])

        return {
            "code_script": code,
            "status": "executing"
        }

    def _node_executor(self, state: ResearchState) -> Dict[str, Any]:
        """节点 4: 沙箱隔离执行"""
        success, logs, metrics, error_tb = SandboxExecutor.run_script(
            code=state["code_script"],
            work_dir=state["work_dir"],
            timeout_seconds=self.timeout_seconds
        )
        retry_count = state.get("retry_count", 0)
        if not success:
            retry_count += 1

        return {
            "execution_success": success,
            "execution_logs": logs,
            "metrics": metrics,
            "error_traceback": error_tb,
            "retry_count": retry_count,
            "status": "executed"
        }

    def _route_after_execution(self, state: ResearchState) -> str:
        """执行后条件路由：成功进入可视化；失败且未超最大重试则自愈修复；超限则兜底推进"""
        if state["execution_success"]:
            return "visualizer"
        if state["retry_count"] < self.max_retries:
            return "coder"
        return "visualizer"

    def _node_visualizer(self, state: ResearchState) -> Dict[str, Any]:
        """节点 5: 论文图表与数据分析"""
        plots, tables = self.visualizer_agent.run(state["metrics"], state["work_dir"])
        return {
            "plots": plots,
            "latex_tables": tables,
            "status": "writing"
        }

    def _node_writer(self, state: ResearchState) -> Dict[str, Any]:
        """节点 6: 学术论文全自动撰写"""
        review_feedback = ""
        rev = state.get("review")
        if rev:
            review_feedback = f"Prior Review Score: {rev.get('overall_score')}. Weaknesses: {rev.get('weaknesses')}. Recommendations: {rev.get('recommendations')}"

        res = self.writer_agent.run(
            idea=state["idea"],
            papers=state["papers"],
            plan=state["experiment_plan"],
            metrics=state["metrics"],
            plots=state["plots"],
            latex_tables=state["latex_tables"],
            review_feedback=review_feedback
        )
        return {
            "paper_sections": res["paper_sections"],
            "paper_bibtex": res["paper_bibtex"],
            "paper_markdown": res["paper_markdown"],
            "paper_latex": res["paper_latex"],
            "status": "reviewing"
        }

    def _node_reviewer(self, state: ResearchState) -> Dict[str, Any]:
        """节点 7: 顶会级模拟同行评审"""
        review = self.reviewer_agent.run(state["paper_markdown"], state["metrics"])
        revision_count = state.get("revision_count", 0)
        overall_score = review.get("overall_score", 0.0)

        if overall_score < self.acceptance_threshold:
            revision_count += 1

        return {
            "review": review,
            "revision_count": revision_count,
            "status": "reviewed"
        }

    def _route_after_review(self, state: ResearchState) -> str:
        """评审后条件路由：评分达标或达修改上限则导出；未达标则触发论文针对性深化"""
        score = state.get("review", {}).get("overall_score", 0.0)
        if score >= self.acceptance_threshold or state.get("revision_count", 0) >= self.max_revisions:
            return "export"
        return "writer"

    def _node_export(self, state: ResearchState) -> Dict[str, Any]:
        """节点 8: 科研成果一键打包导出"""
        ResearchExporter.export_all(
            work_dir=state["work_dir"],
            paper_markdown=state["paper_markdown"],
            paper_latex=state["paper_latex"],
            paper_bibtex=state["paper_bibtex"],
            metrics=state["metrics"],
            review=state["review"],
            plots=state["plots"]
        )
        return {
            "status": "completed"
        }

    def _build_graph(self):
        """构建 LangGraph 状态图"""
        builder = StateGraph(ResearchState)

        # 注册节点
        builder.add_node("ideation", self._node_ideation)
        builder.add_node("planner", self._node_planner)
        builder.add_node("coder", self._node_coder)
        builder.add_node("executor", self._node_executor)
        builder.add_node("visualizer", self._node_visualizer)
        builder.add_node("writer", self._node_writer)
        builder.add_node("reviewer", self._node_reviewer)
        builder.add_node("export", self._node_export)

        # 构建图边
        builder.set_entry_point("ideation")
        builder.add_edge("ideation", "planner")
        builder.add_edge("planner", "coder")
        builder.add_edge("coder", "executor")

        # 执行自愈分支
        builder.add_conditional_edges(
            "executor",
            self._route_after_execution,
            {
                "coder": "coder",
                "visualizer": "visualizer"
            }
        )

        builder.add_edge("visualizer", "writer")
        builder.add_edge("writer", "reviewer")

        # 评审修改分支
        builder.add_conditional_edges(
            "reviewer",
            self._route_after_review,
            {
                "writer": "writer",
                "export": "export"
            }
        )

        builder.add_edge("export", END)

        return builder.compile()
