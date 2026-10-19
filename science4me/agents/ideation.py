import re
from typing import Any, Dict, List, Optional
from science4me.engine.llm import LLMClient
from science4me.tools.academic_search import AcademicSearch


class IdeationAgent:
    """
    选题假说与文献检索智能体 (IdeationAgent)
    7层架构增强落地：
    1. Identity: 顶级实验室主任科学家 (NeurIPS/ICLR Area Chair)，拒绝肤浅调参，专注理论缺口与真问题
    2. Context: 感知算力预算配额、目标顶会等级偏好与领域前沿背景
    3. Skills: 学术检索关键词动态衍生、SOTA 文献痛点挖掘、创新度与可行性多维评估
    4. Memory: 历史课题检索防重记忆，避免重复低效试错
    5. Connections: 输出高标准结构化科研假说与研究问题，向下游 PlannerAgent 输送规划契约
    6. Verification: 假说逻辑完整性断言 (verify_and_refine_idea) 与创新度打分边界约束
    7. Automations: 文献噪声自动清洗、多重查询合并去重与高质量预设假说自适应装配
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def run(
        self,
        topic: str,
        max_papers: int = 5,
        context: Optional[Dict[str, Any]] = None,
        history_topics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """根据用户输入的科研方向，检索 ArXiv 前沿文献并构思可落地的创新假说"""
        ctx = context or {
            "compute_budget": "single_gpu_or_fast_cpu",
            "time_budget_mins": 5,
            "target_tier": "NeurIPS/ICLR"
        }

        # 1. 技能层 (Skills) & 自动化 (Automations): 动态衍生关键词检索与降噪
        search_queries = self._derive_search_queries(topic)
        all_papers = []
        for q in search_queries[:2]:
            papers = AcademicSearch.search_arxiv(q, max_results=max_papers)
            all_papers.extend(papers)

        filtered_papers = self._filter_and_deduplicate_papers(all_papers, topic)
        if not filtered_papers:
            filtered_papers = AcademicSearch.search_arxiv(topic, max_results=max_papers)

        # 2. 假说生成 (LLM / 规则兜底)
        if self.llm.has_real_key:
            idea = self._generate_with_llm(topic, filtered_papers, ctx, history_topics)
        else:
            idea = self._generate_fallback(topic, filtered_papers, ctx)

        # 3. 校验层 (Verification): 审查研究问题完备性、创新分合法性并自愈
        idea = self.verify_and_refine_idea(idea, topic)

        return {
            "papers": filtered_papers,
            "idea": idea,
            "context_used": ctx
        }

    def _derive_search_queries(self, topic: str) -> List[str]:
        """技能层: 动态衍生高针对性文献检索词"""
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
        """自动化层: 过滤摘要过短的噪音文献并按 ArXiv ID 去重"""
        seen_keys = set()
        clean_list = []

        for p in papers:
            bib = p.get("bib_key") or p.get("title", "")
            if not bib or bib in seen_keys:
                continue
            seen_keys.add(bib)

            abstract = p.get("abstract", "")
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
        ]) or "No prior literature fetched. Rely on foundational theoretical machine learning principles."

        history_clause = ""
        if history_topics:
            history_clause = f"\nPreviously explored proposals to avoid duplicating: {history_topics}\n"

        system_prompt = (
            "You are a Principal AI Scientist at a premier AI research lab (DeepMind, FAIR) and Senior Area Chair for NeurIPS/ICLR. "
            "You reject superficial parameter hacking and unprincipled prompt engineering. "
            "Your philosophy is hypothesis-driven, theoretically grounded, and empirically falsifiable. "
            "Analyze the given research topic, hardware context, and retrieved literature: "
            "1. Pinpoint the fundamental technical gap or representation bottleneck in existing literature. "
            "2. Propose a novel, sound, and practically implementable machine learning methodology. "
            "3. Formulate 2-3 precise research questions (RQs) that the experimental plan must answer. "
            "4. Return strictly valid JSON matching the requested schema."
        )

        user_prompt = f"""
Research Topic: {topic}
Runtime Environment: {context}
{history_clause}
Recent Literature:
{lit_summary}

Brainstorm a rigorous, high-impact ML research proposal in JSON format:
{{
  "proposed_title": "string: clear and precise academic paper title",
  "research_questions": [
    "RQ1: precise research question regarding convergence or representation",
    "RQ2: precise research question regarding generalization or ablation"
  ],
  "motivation": "in-depth explanation of why existing methods are inadequate",
  "proposed_method": "core algorithmic mechanism, mathematical formulation, and architecture novelty",
  "expected_contributions": [
    "Contribution 1: novel architectural mechanism",
    "Contribution 2: rigorous empirical validation on standard benchmarks",
    "Contribution 3: extensive ablation studies isolating component contributions"
  ],
  "novelty_score": 8.5,
  "theoretical_foundation": "brief theoretical justification"
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
            "proposed_title": f"Adaptive Dynamic Optimization and Feature Modulation for {topic}",
            "research_questions": [
                f"How does adaptive dynamic feature weighting mitigate representation bottlenecks in {topic}?",
                "Does the proposed residual variance scaling maintain robust convergence without vanishing gradients?"
            ],
            "motivation": f"Traditional architectures for {topic} suffer from rigid static weighting and severe gradient variance decay in high-dimensional feature spaces.",
            "proposed_method": "We formulate an adaptive feature modulation mechanism featuring dynamic gating, normalized variance scaling, and subspace self-gating to dynamically balance information flow.",
            "expected_contributions": [
                f"A novel adaptive modulation mechanism tailored for {topic}.",
                "Comprehensive empirical benchmarking against standard baseline perceptrons demonstrating measurable convergence speedup.",
                "Systematic ablation studies validating the individual necessity of dynamic gating and residual variance scaling."
            ],
            "novelty_score": 8.2,
            "theoretical_foundation": "Based on dynamical systems stability and bounded Lipschitz continuity of residual gating operators."
        }

    def verify_and_refine_idea(self, idea: Dict[str, Any], topic: str) -> Dict[str, Any]:
        """校验层 (Verification): 审查假说字段完整性、打分区间与结构自洽"""
        if not isinstance(idea, dict):
            idea = {}

        if not idea.get("proposed_title"):
            idea["proposed_title"] = f"Empirical Architecture Investigation on {topic}"

        rqs = idea.get("research_questions", [])
        if not isinstance(rqs, list) or len(rqs) < 2:
            idea["research_questions"] = [
                f"How does the proposed architecture impact convergence on {topic}?",
                "Does the component-wise ablation confirm the necessity of the novel modulation mechanism?"
            ]

        if not idea.get("motivation"):
            idea["motivation"] = f"Prior approaches to {topic} suffer from capacity saturation and static feature projection limitations."

        if not idea.get("proposed_method"):
            idea["proposed_method"] = "Dynamic feature gating combined with residual variance scaling."

        contribs = idea.get("expected_contributions", [])
        if not isinstance(contribs, list) or not contribs:
            idea["expected_contributions"] = [
                "Novel adaptive neural architecture formulation.",
                "Rigorous empirical benchmarking against established baselines."
            ]

        try:
            score = float(idea.get("novelty_score", 8.0))
            idea["novelty_score"] = max(1.0, min(10.0, score))
        except (ValueError, TypeError):
            idea["novelty_score"] = 8.0

        return idea
