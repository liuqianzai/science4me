from typing import Any, Dict, List
from science4me.engine.llm import LLMClient
from science4me.tools.academic_search import AcademicSearch


class GoalArchitectureAgent:
    """
    智能体 1: 选题与目标架构智能体
    职责：
    1. 查阅前沿学术文献并提炼学术痛点
    2. 确立明确科研课题
    3. 设定最终量化目标（作为后续训练与演进 Agent 的停止条件）
    4. 规划初步模型架构与演进策略路线
    5. 预估各个架构的初级成果
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def run(self, user_topic: str, max_papers: int = 5) -> Dict[str, Any]:
        """执行文献查阅、选题定义、终止目标设定与架构规划"""
        # 1. 文献查阅
        papers = AcademicSearch.search_arxiv(user_topic, max_results=max_papers)

        # 2. 方案生成
        if self.llm.has_real_key:
            result = self._generate_with_llm(user_topic, papers)
        else:
            result = self._generate_fallback(user_topic, papers)

        result["papers"] = papers
        return result

    def _generate_with_llm(self, topic: str, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        lit_summary = "\n".join([
            f"- [{p.get('bib_key')}] {p.get('title')} ({p.get('published')}): {p.get('abstract')[:160]}..."
            for p in papers
        ])

        system_prompt = (
            "You are a Principal AI Architect and Research Scientist. "
            "Given a research topic and retrieved literature, define: "
            "1. An exact formulated research topic and core problem. "
            "2. A quantitative final stopping target (e.g. test_accuracy >= 0.95 or test_loss <= 0.005) "
            "   that acts as the termination criteria for the autonomous training/optimization loop. "
            "3. The initial model architecture design. "
            "4. A structured multi-generation evolution strategy roadmap (Gen 1 to Gen 3). "
            "5. Expected initial milestone outcomes for each architectural stage. "
            "Return valid JSON matching the exact schema."
        )

        user_prompt = f"""
Research Topic: {topic}

Retrieved Papers:
{lit_summary}

Please formulate the goal and architecture evolution plan in JSON format:
{{
  "topic": "refined clear topic string",
  "target_goal": {{
    "metric": "test_accuracy",
    "threshold": 0.95,
    "condition": ">=",
    "description": "Achieve >= 95.0% test accuracy on standard classification benchmark"
  }},
  "initial_architecture": {{
    "name": "Gen1_BaselineModularNet",
    "core_mechanisms": "Layer normalization and dynamic gating projection",
    "design_rationale": "Overcome static weight decay bottleneck observed in literature"
  }},
  "evolution_strategy": [
    {{"generation": 1, "architecture": "Gen1_BaselineModularNet", "focus": "Establish baseline stability with dynamic gating"}},
    {{"generation": 2, "architecture": "Gen2_AdaptiveVarianceModulator", "focus": "Introduce feature variance scaling and residual modulation"}},
    {{"generation": 3, "architecture": "Gen3_MultiHeadSelfGatedNet", "focus": "Introduce multi-subspace self-gated attention connections"}}
  ],
  "expected_outcomes": {{
    "Gen1": "Attain ~91-92% accuracy with stable convergence",
    "Gen2": "Reach ~93-94% accuracy with reduced gradient variance",
    "Gen3": "Hit the final stopping goal of >=95% accuracy"
  }}
}}
"""
        return self.llm.chat_json([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

    def _generate_fallback(self, topic: str, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "topic": f"Adaptive Dynamic Optimization and Feature Modulation for {topic}",
            "target_goal": {
                "metric": "test_accuracy",
                "threshold": 0.95,
                "condition": ">=",
                "description": "在基准数据集上达到 >= 95.00% 的泛化测试准确率（作为优化终止目标）"
            },
            "initial_architecture": {
                "name": "Gen1_DynamicGatedNet",
                "core_mechanisms": "全连接层 + LayerNorm 稳定化 + Sigmoid 动态特征门控调制",
                "design_rationale": "克服传统静态权重层在训练初期表达能力受限及梯度饱和的问题"
            },
            "evolution_strategy": [
                {
                    "generation": 1,
                    "architecture": "Gen1_DynamicGatedNet",
                    "focus": "建立基线门控结构，实现前馈特征初步自适应缩放"
                },
                {
                    "generation": 2,
                    "architecture": "Gen2_ResidualVarianceModulator",
                    "focus": "引入残差跳跃连接与输入相关方差补偿，增强深层特征传递"
                },
                {
                    "generation": 3,
                    "architecture": "Gen3_MultiHeadSelfGatedNet",
                    "focus": "引入多头子空间自门控机制，达到最终停止目标"
                }
            ],
            "expected_outcomes": {
                "Gen1": "实现基准收敛，预期准确率达 91.0% ~ 93.0%",
                "Gen2": "通过方差补偿降低损失波动，预期准确率提升至 93.5% ~ 94.5%",
                "Gen3": "多头子空间门控消除特征干扰，达成 >= 95.0% 的最终终止目标"
            }
        }
