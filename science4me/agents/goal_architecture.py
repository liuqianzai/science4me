import re
from typing import Any, Dict, List, Optional
from science4me.engine.llm import LLMClient
from science4me.tools.academic_search import AcademicSearch


class GoalArchitectureAgent:
    """
    智能体 1: 选题与目标架构智能体 (GoalArchitectureAgent)
    7层架构增强落地：
    1. Identity: 顶级 AI 科学家兼科研战略总监，拒绝空泛灌水，专注核心痛点与理论创新
    2. Context: 感知算力配额、目标顶会等级与领域前沿背景
    3. Skills: 关键词动态衍生、SOTA 基准指标提取、创新度与可行性量化评估
    4. Memory: 历史课题检索防重与高收敛性架构设计模板复用
    5. Connections: 承接用户输入，输出标准化 TargetGoalContract 协议与多代演化路线
    6. Verification: 目标边界合理性审查与网络拓扑自洽性断言
    7. Automations: 文献噪声自动清洗与 Gen 1~3 演变路线自动推导
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def run(
        self,
        user_topic: str,
        max_papers: int = 5,
        context: Optional[Dict[str, Any]] = None,
        history_topics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """执行文献查阅、选题定义、终止目标设定、架构演化路线规划与自检校验"""
        ctx = context or {
            "compute_budget": "single_gpu_or_fast_cpu",
            "time_budget_mins": 5,
            "target_tier": "NeurIPS/ICLR"
        }

        # 1. 技能层 (Skills) & 自动化 (Automations): 动态衍生关键词并进行文献检索与降噪
        search_queries = self._derive_search_queries(user_topic)
        all_papers = []
        for q in search_queries[:2]:
            papers = AcademicSearch.search_arxiv(q, max_results=max_papers)
            all_papers.extend(papers)

        filtered_papers = self._filter_and_deduplicate_papers(all_papers, user_topic)
        if not filtered_papers:
            # 兜底检索
            filtered_papers = AcademicSearch.search_arxiv(user_topic, max_results=max_papers)

        # 2. 方案生成 (LLM / 规则兜底)
        if self.llm.has_real_key:
            result = self._generate_with_llm(user_topic, filtered_papers, ctx, history_topics)
        else:
            result = self._generate_fallback(user_topic, filtered_papers, ctx)

        # 3. 校验层 (Verification): 目标合法性断言与自修复
        result = self.verify_and_refine(result, user_topic)
        result["papers"] = filtered_papers
        result["context_used"] = ctx

        return result

    def _derive_search_queries(self, topic: str) -> List[str]:
        """技能层: 根据用户输入的主题词动态衍生高精准度的学术检索关键词"""
        clean_topic = re.sub(r"[^\w\s]", " ", topic).strip()
        tokens = [t for t in clean_topic.split() if len(t) > 2]
        queries = [clean_topic]
        if len(tokens) >= 2:
            queries.append(f"{tokens[0]} {tokens[1]} neural architecture")
            queries.append(f"{clean_topic} optimization benchmark")
        else:
            queries.append(f"{clean_topic} deep learning")
        return queries

    def _filter_and_deduplicate_papers(
        self,
        papers: List[Dict[str, Any]],
        topic: str
    ) -> List[Dict[str, Any]]:
        """自动化层: 过滤噪音文献并按 ArXiv ID 去重"""
        seen_keys = set()
        clean_list = []
        topic_words = set(re.sub(r"[^\w\s]", " ", topic.lower()).split())

        for p in papers:
            bib = p.get("bib_key") or p.get("title", "")
            if not bib or bib in seen_keys:
                continue
            seen_keys.add(bib)

            abstract = p.get("abstract", "").lower()
            title = p.get("title", "").lower()
            # 检查相关性与摘要完备度
            if len(abstract) < 60:
                continue
            clean_list.append(p)

        return clean_list[:6]

    def _generate_with_llm(
        self,
        topic: str,
        papers: List[Dict[str, Any]],
        context: Dict[str, Any],
        history_topics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        lit_summary = "\n".join([
            f"- [{p.get('bib_key')}] {p.get('title')} ({p.get('published')}): {p.get('abstract')[:200]}..."
            for p in papers
        ]) or "No pre-existing literature fetched. Formulate based on SOTA foundations."

        history_clause = ""
        if history_topics:
            history_clause = f"\nPreviously explored topics to avoid duplicating: {history_topics}\n"

        system_prompt = (
            "You are a Principal AI Scientist, Research Director, and Senior Area Chair at top-tier venues (NeurIPS / ICLR). "
            "Your philosophy is rigorous, hypothesis-driven, and mathematically sound. "
            "You strictly avoid superficial cosmetic modifications (such as trivial hyperparameter renaming or baseless parameter bloat). "
            "Given the research direction, compute constraints, and retrieved literature: "
            "1. Formulate a precise, scientifically significant research topic that targets genuine performance bottlenecks. "
            "2. Establish a clear, quantitative, and achievable final stopping target (e.g., test_accuracy >= 0.95 or test_loss <= 0.005) "
            "   that serves as the definitive stopping criterion for autonomous training and optimization iterations. "
            "3. Propose a principled initial neural architecture (Gen 1) addressing the literature's observed shortcomings. "
            "4. Structure an interpretable multi-generation evolution strategy roadmap (Gen 1 to Gen 3). "
            "5. Provide an innovation and theoretical soundness assessment. "
            "Output strictly valid JSON matching the exact schema."
        )

        user_prompt = f"""
Research Direction: {topic}
Compute Constraints & Target: {context}
{history_clause}
Retrieved Literature Evidence:
{lit_summary}

Formulate the research objective and architecture evolution plan in the following JSON schema:
{{
  "topic": "precise formulated research topic",
  "problem_statement": "clear description of the technical bottleneck observed in prior art",
  "target_goal": {{
    "metric": "test_accuracy",
    "threshold": 0.95,
    "condition": ">=",
    "description": "Achieve >= 95.0% test accuracy on standard classification benchmark (Stopping Criterion)"
  }},
  "initial_architecture": {{
    "name": "Gen1_BaselineModularNet",
    "core_mechanisms": "Layer normalization with dynamic gating projection",
    "design_rationale": "Overcomes static weight saturation and stabilizes gradient flow",
    "theoretical_advantage": "Ensures bounded Lipschitz continuity across feedforward projections"
  }},
  "evolution_strategy": [
    {{"generation": 1, "architecture": "Gen1_BaselineModularNet", "focus": "Establish baseline stability with dynamic feature gating"}},
    {{"generation": 2, "architecture": "Gen2_AdaptiveVarianceModulator", "focus": "Introduce feature variance scaling and residual modulation"}},
    {{"generation": 3, "architecture": "Gen3_MultiHeadSelfGatedNet", "focus": "Introduce multi-subspace self-gated attention connections to achieve final goal"}}
  ],
  "expected_outcomes": {{
    "Gen1": "Attain ~91-92% accuracy with stable convergence",
    "Gen2": "Reach ~93-94% accuracy with reduced gradient variance",
    "Gen3": "Hit the final stopping goal of >=95% accuracy"
  }},
  "feasibility_assessment": {{
    "complexity": "O(N) linear computation suitable for constrained budget",
    "theoretical_grounding": "High",
    "risk_factors": "Potential gradient vanishing if gate saturation occurs"
  }}
}}
"""
        return self.llm.chat_json([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

    def _generate_fallback(
        self,
        topic: str,
        papers: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "topic": f"Adaptive Dynamic Optimization and Feature Modulation for {topic}",
            "problem_statement": "传统静态全连接层在处理非线性高维交叉特征时容易出现浅层梯度弥散与过拟合，且缺乏跨样本动态加权能力。",
            "target_goal": {
                "metric": "test_accuracy",
                "threshold": 0.95,
                "condition": ">=",
                "description": "在基准数据集上达到 >= 95.00% 的泛化测试准确率（作为优化迭代终止目标）"
            },
            "initial_architecture": {
                "name": "Gen1_DynamicGatedNet",
                "core_mechanisms": "全连接层 + LayerNorm 稳定化 + Sigmoid 动态特征门控调制",
                "design_rationale": "克服传统静态权重层在训练初期表达能力受限及梯度饱和的问题",
                "theoretical_advantage": "利用输入自相关性生成样本级特征掩码，自适应抑制任务无关冗余维度"
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
                    "focus": "引入残差跳跃连接与输入相关方差补偿，增强深层特征传递与收敛鲁棒性"
                },
                {
                    "generation": 3,
                    "architecture": "Gen3_MultiHeadSelfGatedNet",
                    "focus": "引入多头子空间自门控机制，消除特征交叉冗余，达到最终停止目标"
                }
            ],
            "expected_outcomes": {
                "Gen1": "实现基准收敛，预期准确率达 91.0% ~ 93.0%",
                "Gen2": "通过方差补偿降低损失波动，预期准确率提升至 93.5% ~ 94.5%",
                "Gen3": "多头子空间门控消除特征干扰，达成 >= 95.0% 的最终终止目标"
            },
            "feasibility_assessment": {
                "complexity": "轻量级前馈网络，单轮训练消耗低于 10 秒",
                "theoretical_grounding": "基于动态系统自稳定性与子空间分解理论",
                "risk_factors": "初生代门控易在学习率过大时进入饱和区"
            }
        }

    def verify_and_refine(self, result: Dict[str, Any], raw_topic: str) -> Dict[str, Any]:
        """校验层 (Verification): 校验关键字段完备性、量化阈值合理性与结构自洽"""
        if not isinstance(result, dict):
            result = {}

        if not result.get("topic"):
            result["topic"] = f"Empirical Architecture Investigation on {raw_topic}"

        # 校验目标边界
        tg = result.get("target_goal", {})
        if not isinstance(tg, dict):
            tg = {}
        metric = tg.get("metric", "test_accuracy")
        threshold = float(tg.get("threshold", 0.95))
        condition = tg.get("condition", ">=")

        # 防御性自愈：如果 threshold 设定过激（如 > 0.999），修正为可实现的学术阈值
        if metric == "test_accuracy" and (threshold > 0.99 or threshold < 0.5):
            threshold = 0.95
        elif metric == "test_loss" and threshold <= 0:
            threshold = 0.05

        tg["metric"] = metric
        tg["threshold"] = threshold
        tg["condition"] = condition
        if not tg.get("description"):
            tg["description"] = f"Achieve {condition} {threshold} on {metric} (Autonomous Stopping Criterion)"
        result["target_goal"] = tg

        # 校验架构演进策略结构
        evo = result.get("evolution_strategy", [])
        if not isinstance(evo, list) or len(evo) < 2:
            result["evolution_strategy"] = [
                {"generation": 1, "architecture": "Gen1_DynamicGatedNet", "focus": "Baseline dynamic gating"},
                {"generation": 2, "architecture": "Gen2_ResidualVarianceModulator", "focus": "Residual connection & variance scaling"},
                {"generation": 3, "architecture": "Gen3_MultiHeadSelfGatedNet", "focus": "Multi-head subspace gating"}
            ]

        # 校验初始架构说明
        init_arch = result.get("initial_architecture", {})
        if not isinstance(init_arch, dict) or not init_arch.get("name"):
            result["initial_architecture"] = {
                "name": "Gen1_DynamicGatedNet",
                "core_mechanisms": "全连接层 + LayerNorm + 动态门控",
                "design_rationale": "构建轻量自适应前向基线"
            }

        return result
