import json
from typing import Any, Dict, List
from science4me.engine.llm import LLMClient
from science4me.tools.academic_search import AcademicSearch


class IdeationAgent:
    """选题假说与文献检索智能体"""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def run(self, topic: str, max_papers: int = 5) -> Dict[str, Any]:
        """
        根据用户输入的科研方向，检索 ArXiv 前沿文献并构思创新假说
        """
        # 1. 自动检索相关文献
        papers = AcademicSearch.search_arxiv(topic, max_results=max_papers)

        # 2. 生成创新假说
        if self.llm.has_real_key:
            idea = self._generate_with_llm(topic, papers)
        else:
            idea = self._generate_fallback(topic, papers)

        return {
            "papers": papers,
            "idea": idea
        }

    def _generate_with_llm(self, topic: str, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        lit_summary = "\n".join([
            f"- [{p.get('bib_key')}] {p.get('title')} ({p.get('published')})\n  Abstract: {p.get('abstract')[:200]}..."
            for p in papers
        ])

        system_prompt = (
            "You are a Principal AI Scientist at a top AI research lab (e.g., DeepMind, FAIR). "
            "Your task is to analyze the research topic and existing literature, identify research gaps, "
            "and propose a novel, sound, and practically implementable machine learning hypothesis. "
            "Return valid JSON strictly matching the requested format."
        )

        user_prompt = f"""
Research Topic: {topic}

Recent Literature:
{lit_summary}

Please brainstorm a solid, novel ML research proposal. Return a JSON object with:
{{
  "proposed_title": "string: academic paper title",
  "research_questions": ["string: RQ1", "string: RQ2"],
  "motivation": "string: why existing methods are inadequate",
  "proposed_method": "string: core algorithmic idea, formulation, and novelty",
  "expected_contributions": ["string: contribution 1", "string: contribution 2"],
  "novelty_score": 8.5
}}
"""
        return self.llm.chat_json([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

    def _generate_fallback(self, topic: str, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "proposed_title": f"Adaptive Dynamic Optimization and Feature Modulation for {topic}",
            "research_questions": [
                f"How does adaptive feature weighting impact convergence and generalization in {topic}?",
                "Does the proposed method maintain robust performance across varied training noisy regimes?"
            ],
            "motivation": f"Traditional architectures and baselines for {topic} suffer from rigid static weighting and slow convergence during early training phases.",
            "proposed_method": "We introduce an adaptive dynamic modulation layer with normalized attention weighting that dynamically scales gradient variance during training.",
            "expected_contributions": [
                f"A novel adaptive modulation mechanism specifically designed for {topic}.",
                "Comprehensive empirical benchmarking against standard baselines demonstrating superior convergence and stability.",
                "Ablation studies validating each component of the dynamic scaling function."
            ],
            "novelty_score": 8.0
        }
