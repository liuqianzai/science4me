from typing import Any, Dict, List, Optional
import torch
from science4me.engine.llm import LLMClient


class PlannerAgent:
    """
    实验规划与评测基线设计智能体 (PlannerAgent)
    7层架构增强落地：
    1. Identity: 资深机器学习实验计量学家，恪守单一变量原则与可复现性规范
    2. Context: 动态感知宿主机硬件后端 (CUDA/CPU) 与训练时限预算
    3. Skills: 自动生成消融实验对照组矩阵、复合评估指标体系、稳定优化器参数配置
    4. Memory: 经典深度网络超参黄金收敛区间知识沉淀
    5. Connections: 承接 IdeationAgent/GoalArchitectureAgent 假说输入，向下游 CoderAgent 输出严谨的规范协议
    6. Verification: 实验参数边界断言与变量隔离自洽性审查 (verify_and_refine_plan)
    7. Automations: 算力自适应分配、Epochs与BatchSize自平衡、自动化稳健降级
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def run(
        self,
        idea: Dict[str, Any],
        runtime_env: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """根据创新假说制定可落地、自包含且快速收敛的实验规划方案"""
        env = runtime_env or {
            "cuda_available": torch.cuda.is_available(),
            "recommended_device": "cuda" if torch.cuda.is_available() else "cpu",
            "time_budget_mins": 5
        }

        # 1. 方案生成 (LLM / 规则兜底)
        if self.llm.has_real_key:
            plan = self._plan_with_llm(idea, env)
        else:
            plan = self._plan_fallback(idea, env)

        # 2. 校验层 (Verification): 审查方案完整性、超参安全边界并自愈
        plan = self.verify_and_refine_plan(plan, idea, env)
        plan["runtime_env"] = env

        return plan

    def _plan_with_llm(self, idea: Dict[str, Any], env: Dict[str, Any]) -> Dict[str, Any]:
        system_prompt = (
            "You are a Senior Machine Learning Research Engineer and Experimental Metrology Specialist. "
            "Your highest principle is the 'Control Variate Principle' and scientific reproducibility. "
            "Design a rigorous, realistic, and self-contained experiment plan to validate the proposed hypothesis. "
            "Ensure the experiment: "
            "1. Specifies conventional standard baselines for fair, unbiased benchmarking. "
            "2. Defines explicit ablation variants (e.g. without dynamic gating, without LayerNorm) to isolate mechanism contributions. "
            "3. Uses standard datasets (torchvision or scikit-learn standard benchmarks) that converge locally within minutes. "
            "4. Configures defensive hyperparameters to avoid divergence (e.g. AdamW with Cosine Annealing). "
            "Return valid JSON strictly matching the requested format."
        )

        user_prompt = f"""
Research Proposal:
Title: {idea.get('proposed_title')}
Motivation: {idea.get('motivation')}
Method: {idea.get('proposed_method')}
Research Questions: {idea.get('research_questions')}
Runtime Environment: {env}

Design an experiment plan in JSON format:
{{
  "benchmark_task": "string: task description and dataset",
  "dataset_source": "string: synthetic_multiclass or torchvision standard dataset",
  "dataset_config": {{
    "samples": 3000,
    "features": 32,
    "classes": 3,
    "test_split": 0.25,
    "random_state": 42
  }},
  "evaluation_metrics": [
    "test_accuracy",
    "test_loss",
    "f1_score",
    "convergence_epoch",
    "training_time_sec"
  ],
  "baselines": [
    {{"name": "StandardBaseline", "description": "Conventional standard model architecture"}},
    {{"name": "StaticModulation", "description": "Baseline with fixed non-dynamic weights"}}
  ],
  "proposed_model": {{
    "name": "ProposedAdaptiveModel",
    "description": "Full novel proposed architecture implementing dynamic gating"
  }},
  "ablations": [
    {{"name": "Ablation_NoNorm", "description": "Proposed model without normalization layer"}},
    {{"name": "Ablation_NoDynamicGate", "description": "Proposed model without dynamic gating"}}
  ],
  "hyperparameters": {{
    "epochs": 15,
    "batch_size": 64,
    "learning_rate": 0.002,
    "weight_decay": 0.0001,
    "optimizer": "AdamW",
    "lr_scheduler": "CosineAnnealingLR",
    "device_preference": "{env.get('recommended_device', 'cpu')}"
  }},
  "expected_plots": [
    "loss_curves.png",
    "benchmark_comparison.png",
    "pareto_efficiency.png"
  ]
}}
"""
        return self.llm.chat_json([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

    def _plan_fallback(self, idea: Dict[str, Any], env: Dict[str, Any]) -> Dict[str, Any]:
        device = env.get("recommended_device", "cpu")
        return {
            "benchmark_task": "高维非线性特征分类基准评测 (支持 CUDA 硬件加速)",
            "dataset_source": "synthetic_multiclass",
            "dataset_config": {
                "samples": 3000,
                "features": 32,
                "informative_features": 24,
                "classes": 3,
                "test_split": 0.25,
                "random_state": 42
            },
            "evaluation_metrics": [
                "test_accuracy",
                "test_loss",
                "f1_score",
                "convergence_epoch",
                "training_time_sec"
            ],
            "baselines": [
                {"name": "StandardBaseline", "description": "传统标准全连接多层感知机 (静态线性投射基线)"},
                {"name": "StaticModulation", "description": "静态固定加权缩放感知机 (非自适应对照组)"}
            ],
            "proposed_model": {
                "name": "ProposedAdaptiveModel",
                "description": "引入动态特征门控与自适应方差调制的完整创新模型"
            },
            "ablations": [
                {"name": "Ablation_NoDynamicGate", "description": "剥离动态门控机制的直接前馈消融组"},
                {"name": "Ablation_NoNorm", "description": "剥离 LayerNorm 归一化的消融对照组"}
            ],
            "hyperparameters": {
                "epochs": 15,
                "batch_size": 64,
                "learning_rate": 0.002,
                "weight_decay": 0.0001,
                "optimizer": "AdamW",
                "lr_scheduler": "CosineAnnealingLR",
                "device_preference": device
            },
            "expected_plots": [
                "loss_curves.png",
                "benchmark_comparison.png",
                "pareto_efficiency.png"
            ]
        }

    def verify_and_refine_plan(
        self,
        plan: Dict[str, Any],
        idea: Dict[str, Any],
        env: Dict[str, Any]
    ) -> Dict[str, Any]:
        """校验层 (Verification): 审查实验完备性、超参安全边界与指标闭环"""
        if not isinstance(plan, dict):
            plan = {}

        # 1. 确保评估指标完备
        metrics = plan.get("evaluation_metrics", [])
        if not isinstance(metrics, list) or not metrics:
            metrics = ["test_accuracy", "test_loss", "convergence_epoch", "training_time_sec"]
        if "test_accuracy" not in metrics:
            metrics.insert(0, "test_accuracy")
        plan["evaluation_metrics"] = metrics

        # 2. 确保对照基线与消融组
        baselines = plan.get("baselines", [])
        if not isinstance(baselines, list) or not baselines:
            plan["baselines"] = [
                {"name": "StandardBaseline", "description": "标准静态基线感知机"}
            ]

        ablations = plan.get("ablations", [])
        if not isinstance(ablations, list) or not ablations:
            plan["ablations"] = [
                {"name": "Ablation_NoGate", "description": "无动态门控机制消融组"}
            ]

        # 3. 超参数安全区间约束
        hyp = plan.get("hyperparameters", {})
        if not isinstance(hyp, dict):
            hyp = {}

        epochs = int(hyp.get("epochs", 15))
        hyp["epochs"] = max(5, min(30, epochs))

        bs = int(hyp.get("batch_size", 64))
        hyp["batch_size"] = bs if bs in [16, 32, 64, 128, 256] else 64

        lr = float(hyp.get("learning_rate", 0.002))
        hyp["learning_rate"] = lr if (0.0001 <= lr <= 0.05) else 0.002

        if not hyp.get("device_preference"):
            hyp["device_preference"] = env.get("recommended_device", "cpu")

        plan["hyperparameters"] = hyp

        # 4. 预期图表列表
        plots = plan.get("expected_plots", [])
        if not isinstance(plots, list) or not plots:
            plan["expected_plots"] = ["loss_curves.png", "benchmark_comparison.png", "pareto_efficiency.png"]

        return plan
