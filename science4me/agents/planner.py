from typing import Any, Dict
from science4me.engine.llm import LLMClient


class PlannerAgent:
    """实验规划与评测基线设计智能体"""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def run(self, idea: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据创新假说制定可落地、自包含且快速收敛的实验规划方案
        """
        if self.llm.has_real_key:
            return self._plan_with_llm(idea)
        else:
            return self._plan_fallback(idea)

    def _plan_with_llm(self, idea: Dict[str, Any]) -> Dict[str, Any]:
        system_prompt = (
            "You are a Senior Machine Learning Research Engineer. "
            "Design a rigorous, realistic, and self-contained experiment plan to validate the proposed hypothesis. "
            "Ensure the experiment uses standard datasets (e.g., PyTorch torchvision or scikit-learn synthetic/standard benchmarks) "
            "that can run and converge locally within a few minutes. "
            "Return valid JSON strictly matching the requested format."
        )

        user_prompt = f"""
Proposed Research:
Title: {idea.get('proposed_title')}
Motivation: {idea.get('motivation')}
Method: {idea.get('proposed_method')}
Research Questions: {idea.get('research_questions')}

Design an experiment plan. Return a JSON object with:
{{
  "benchmark_task": "string: task description and dataset",
  "dataset_source": "string: torchvision or sklearn",
  "evaluation_metrics": ["accuracy", "loss", "convergence_epoch", "training_time_sec"],
  "baselines": [
    {{"name": "StandardBaseline", "description": "Conventional standard model architecture"}},
    {{"name": "StaticModulation", "description": "Baseline with fixed non-dynamic weights"}}
  ],
  "proposed_model": {{"name": "ProposedAdaptiveModel", "description": "Full novel proposed architecture"}},
  "ablations": [
    {{"name": "Proposed_NoNorm", "description": "Proposed model without normalization layer"}},
    {{"name": "Proposed_NoDynamicWeight", "description": "Proposed model without dynamic scaling factor"}}
  ],
  "hyperparameters": {{
    "epochs": 10,
    "batch_size": 64,
    "learning_rate": 0.001,
    "seed": 42
  }},
  "expected_plots": ["loss_curves.png", "benchmark_comparison.png", "ablation_study.png"]
}}
"""
        return self.llm.chat_json([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

    def _plan_fallback(self, idea: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "benchmark_task": "Classification Benchmark on Structured Feature Space",
            "dataset_source": "sklearn_synthetic_multiclass",
            "evaluation_metrics": ["accuracy", "val_loss", "convergence_epoch", "training_time_sec"],
            "baselines": [
                {"name": "Baseline_StandardMLP", "description": "Standard multi-layer perceptron with static linear projections"},
                {"name": "Baseline_StaticWeighting", "description": "MLP with fixed non-adaptive weight scaling"}
            ],
            "proposed_model": {
                "name": "Proposed_AdaptiveModulator",
                "description": "Novel architecture incorporating dynamic gating and adaptive feature variance modulation"
            },
            "ablations": [
                {"name": "Ablation_WithoutModulation", "description": "Proposed architecture with dynamic modulation disabled"},
                {"name": "Ablation_WithoutLayerNorm", "description": "Proposed architecture without stabilizing normalization"}
            ],
            "hyperparameters": {
                "epochs": 15,
                "batch_size": 64,
                "learning_rate": 0.002,
                "hidden_dim": 128,
                "seed": 42
            },
            "expected_plots": ["loss_curves.png", "benchmark_comparison.png", "ablation_study.png"]
        }
