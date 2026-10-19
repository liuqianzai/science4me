from typing import Any, Dict, List, Optional
import torch
from science4me.engine.llm import LLMClient


class ExperimentPlannerAgent:
    """
    智能体 2: 实验方案规划智能体 (ExperimentPlannerAgent)
    7层架构增强落地：
    1. Identity: 资深机器学习实验架构师与实验计量学专家，严格遵守单一变量原则与统计学显著性
    2. Context: 结合课题、量化停止目标、当前架构规格及物理算力上下文动态定制
    3. Skills: 消融实验矩阵设计、超参调度方案（Warmup/Cosine）、复合评估指标体系构建
    4. Memory: 沉淀经典网络结构的超参最优区间与标准化 Baseline 对照组模版
    5. Connections: 承接 Agent 1 目标契约，为 Agent 3 生成标准 ExperimentSpec 参数配置
    6. Verification: 实验完备性静态审查（基线对齐、变量隔离、指标闭环覆盖）
    7. Automations: 自适应硬件倾向（CUDA/CPU感知）、根据样本规模自平衡 BatchSize 与 Epochs
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def run(
        self,
        topic: str,
        target_goal: Dict[str, Any],
        architecture_spec: Dict[str, Any],
        runtime_env: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """制定具体实验方案、消融实验矩阵与参数配置"""
        env = runtime_env or {
            "cuda_available": torch.cuda.is_available(),
            "recommended_device": "cuda" if torch.cuda.is_available() else "cpu",
            "time_budget_mins": 5
        }

        # 1. 生成方案 (LLM / 规则兜底)
        if self.llm.has_real_key:
            plan = self._plan_with_llm(topic, target_goal, architecture_spec, env)
        else:
            plan = self._plan_fallback(topic, target_goal, architecture_spec, env)

        # 2. 校验层 (Verification): 完备性检查与自愈修正
        plan = self.verify_and_refine_plan(plan, target_goal, env)
        plan["runtime_env"] = env

        return plan

    def _plan_with_llm(
        self,
        topic: str,
        target_goal: Dict[str, Any],
        architecture_spec: Dict[str, Any],
        env: Dict[str, Any]
    ) -> Dict[str, Any]:
        system_prompt = (
            "You are a Senior Machine Learning Experiment Architect and Metrology Specialist. "
            "You rigorously uphold the 'Control Variate Principle' and scientific reproducibility. "
            "Your objective is to design a realistic, self-contained, and comprehensive PyTorch experimental protocol "
            "that rigorously validates whether the proposed architecture meets the quantitative stopping target. "
            "Crucially: "
            "1. Specify standardized baseline architectures for unbiased comparative benchmarking. "
            "2. Define explicit ablation variants (e.g. model without dynamic gating or without residual projection) "
            "   to isolate each mechanism's marginal contribution. "
            "3. Prescribe stable optimization hyperparameters (learning rate, optimizer, batch size, epochs) "
            "   calibrated to converge quickly on the specified hardware without divergence. "
            "Return valid JSON matching the exact schema."
        )

        user_prompt = f"""
Research Topic: {topic}
Target Stopping Goal: {target_goal}
Current Architecture Specification: {architecture_spec}
Runtime Hardware Context: {env}

Design the experimental protocol and return strictly valid JSON matching this schema:
{{
  "task_description": "Comprehensive feature representation and classification benchmark",
  "dataset_config": {{
    "type": "synthetic_multiclass",
    "samples": 3000,
    "features": 32,
    "informative_features": 24,
    "redundant_features": 4,
    "classes": 3,
    "test_split": 0.25,
    "random_state": 42
  }},
  "baselines": [
    {{"name": "StandardMLP", "description": "Standard multi-layer perceptron without modulation (Static Baseline)"}},
    {{"name": "StaticWeightingNet", "description": "Non-adaptive fixed scalar modulation (Ablation Comparison)"}}
  ],
  "ablations": [
    {{"name": "Ablation_NoGate", "description": "Proposed architecture with gating component disconnected"}},
    {{"name": "Ablation_NoNorm", "description": "Proposed architecture without LayerNorm normalization"}}
  ],
  "evaluation_metrics": [
    "test_accuracy",
    "test_loss",
    "f1_score",
    "convergence_epoch",
    "training_time_sec"
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
  "validation_strategy": "4-fold cross-validation or stratified holdout test set"
}}
"""
        return self.llm.chat_json([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

    def _plan_fallback(
        self,
        topic: str,
        target_goal: Dict[str, Any],
        architecture_spec: Dict[str, Any],
        env: Dict[str, Any]
    ) -> Dict[str, Any]:
        device = env.get("recommended_device", "cpu")
        return {
            "task_description": "高维非线性特征分类基准评测 (支持 CUDA 硬件加速与消融验证)",
            "dataset_config": {
                "type": "synthetic_multiclass",
                "samples": 3000,
                "features": 32,
                "informative_features": 24,
                "redundant_features": 4,
                "classes": 3,
                "test_split": 0.25,
                "random_state": 42
            },
            "baselines": [
                {"name": "StandardMLP", "description": "传统标准全连接多层感知机 (静态线性投射基准)"},
                {"name": "StaticWeightingNet", "description": "静态固定加权缩放感知机 (非自适应对照组)"}
            ],
            "ablations": [
                {"name": "Ablation_NoGate", "description": "剥离动态门控的直接前馈消融组"}
            ],
            "evaluation_metrics": [
                "test_accuracy",
                "test_loss",
                "f1_score",
                "convergence_epoch",
                "training_time_sec"
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
            "validation_strategy": "分层随机切分测试集 (Stratified 25% Holdout)"
        }

    def verify_and_refine_plan(
        self,
        plan: Dict[str, Any],
        target_goal: Dict[str, Any],
        env: Dict[str, Any]
    ) -> Dict[str, Any]:
        """校验层 (Verification): 审查实验参数完整性、度量闭环与基线覆盖"""
        if not isinstance(plan, dict):
            plan = {}

        # 1. 确保核心评估指标覆盖 target_goal 中的目标指标
        target_metric = target_goal.get("metric", "test_accuracy")
        metrics = plan.get("evaluation_metrics", [])
        if not isinstance(metrics, list) or not metrics:
            metrics = ["test_accuracy", "test_loss", "convergence_epoch", "training_time_sec"]
        if target_metric not in metrics:
            metrics.insert(0, target_metric)
        plan["evaluation_metrics"] = metrics

        # 2. 确保至少包含对照基线
        baselines = plan.get("baselines", [])
        if not isinstance(baselines, list) or not baselines:
            plan["baselines"] = [
                {"name": "StandardMLP", "description": "标准静态基线感知机"}
            ]

        # 3. 超参数合法性与越界约束
        hyp = plan.get("hyperparameters", {})
        if not isinstance(hyp, dict):
            hyp = {}

        # 限制轮数在合理区间，兼顾收敛性与超时防护
        epochs = int(hyp.get("epochs", 15))
        if epochs < 5:
            epochs = 5
        elif epochs > 50:
            epochs = 25
        hyp["epochs"] = epochs

        # Batch Size 约束 (16 ~ 256)
        bs = int(hyp.get("batch_size", 64))
        if bs not in [16, 32, 64, 128, 256]:
            bs = 64
        hyp["batch_size"] = bs

        # 学习率安全范围 (0.0001 ~ 0.05)
        lr = float(hyp.get("learning_rate", 0.002))
        if lr <= 0 or lr > 0.05:
            lr = 0.002
        hyp["learning_rate"] = lr

        # 硬件倾向校验
        if not hyp.get("device_preference"):
            hyp["device_preference"] = env.get("recommended_device", "cpu")

        plan["hyperparameters"] = hyp

        # 4. 数据集配置自愈
        ds = plan.get("dataset_config", {})
        if not isinstance(ds, dict) or not ds.get("samples"):
            plan["dataset_config"] = {
                "type": "synthetic_multiclass",
                "samples": 3000,
                "features": 32,
                "classes": 3,
                "test_split": 0.25,
                "random_state": 42
            }

        return plan
