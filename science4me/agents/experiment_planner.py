from typing import Any, Dict
from science4me.engine.llm import LLMClient


class ExperimentPlannerAgent:
    """
    智能体 2: 实验方案规划智能体
    职责：结合检索文献、选题题目与设定的终止目标，制定严密可落地的具体实验方案
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def run(
        self,
        topic: str,
        target_goal: Dict[str, Any],
        architecture_spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """制定具体实验方案与参数配置"""
        if self.llm.has_real_key:
            return self._plan_with_llm(topic, target_goal, architecture_spec)
        else:
            return self._plan_fallback(topic, target_goal, architecture_spec)

    def _plan_with_llm(
        self,
        topic: str,
        target_goal: Dict[str, Any],
        architecture_spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        system_prompt = (
            "You are a Senior Machine Learning Experiment Planner. "
            "Based on the research goal, stopping target, and proposed architecture, "
            "design a rigorous experimental protocol. Ensure the benchmark dataset, "
            "baselines, and hyperparameters are clearly specified for PyTorch execution. "
            "Return valid JSON matching the schema."
        )

        user_prompt = f"""
Topic: {topic}
Target Stopping Goal: {target_goal}
Current Architecture: {architecture_spec}

Formulate the experiment plan in JSON format:
{{
  "task_description": "High-dimensional feature classification benchmark",
  "dataset_config": {{
    "type": "synthetic_multiclass",
    "samples": 3000,
    "features": 32,
    "classes": 3,
    "test_split": 0.25,
    "random_state": 42
  }},
  "baselines": [
    {{"name": "StandardMLP", "description": "Static multi-layer perceptron without modulation"}},
    {{"name": "StaticWeightingNet", "description": "Fixed scalar modulation without dynamic gating"}}
  ],
  "evaluation_metrics": ["test_accuracy", "test_loss", "convergence_epoch", "training_time_sec"],
  "hyperparameters": {{
    "epochs": 15,
    "batch_size": 64,
    "learning_rate": 0.002,
    "weight_decay": 0.0001,
    "device_preference": "cuda"
  }}
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
        architecture_spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "task_description": "高维非线性特征分类基准评测 (支持 CUDA 硬件加速)",
            "dataset_config": {
                "type": "synthetic_multiclass",
                "samples": 3000,
                "features": 32,
                "informative_features": 24,
                "classes": 3,
                "test_split": 0.25,
                "random_state": 42
            },
            "baselines": [
                {"name": "StandardMLP", "description": "传统标准全连接多层感知机 (静态线性投射)"},
                {"name": "StaticWeightingNet", "description": "静态固定加权缩放感知机 (非自适应)"}
            ],
            "evaluation_metrics": ["test_accuracy", "test_loss", "convergence_epoch", "training_time_sec"],
            "hyperparameters": {
                "epochs": 15,
                "batch_size": 64,
                "learning_rate": 0.002,
                "weight_decay": 0.0001,
                "device_preference": "cuda"
            }
        }
