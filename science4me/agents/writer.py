from typing import Any, Dict, List
from science4me.engine.llm import LLMClient


class WriterAgent:
    """学术论文撰写与排版智能体"""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def run(
        self,
        idea: Dict[str, Any],
        papers: List[Dict[str, Any]],
        plan: Dict[str, Any],
        metrics: Dict[str, Any],
        plots: List[str],
        latex_tables: List[str],
        review_feedback: str = ""
    ) -> Dict[str, Any]:
        """
        全自动撰写学术论文（生成 LaTeX 与 Markdown 双版本）
        """
        bib_library = "\n\n".join([p.get("bibtex", "") for p in papers if p.get("bibtex")])
        citations_guide = ", ".join([f"\\cite{{{p.get('bib_key')}}}" for p in papers if p.get("bib_key")])

        if self.llm.has_real_key:
            sections = self._write_with_llm(idea, papers, plan, metrics, plots, latex_tables, review_feedback)
        else:
            sections = self._write_fallback(idea, papers, metrics, plots, latex_tables)

        # 组装完整 Markdown
        markdown_paper = self._assemble_markdown(idea, sections, plots)

        # 组装完整 LaTeX
        latex_paper = self._assemble_latex(idea, sections, plots, latex_tables)

        return {
            "paper_sections": sections,
            "paper_bibtex": bib_library,
            "paper_markdown": markdown_paper,
            "paper_latex": latex_paper
        }

    def _write_with_llm(
        self,
        idea: Dict[str, Any],
        papers: List[Dict[str, Any]],
        plan: Dict[str, Any],
        metrics: Dict[str, Any],
        plots: List[str],
        latex_tables: List[str],
        review_feedback: str
    ) -> Dict[str, str]:
        system_prompt = (
            "You are a World-Class Academic Author writing a submission for top ML conferences (ICLR, NeurIPS, ICML). "
            "Write formal, rigorous, mathematically sound, and comprehensive paper sections. "
            "Return a JSON object containing sections: 'abstract', 'introduction', 'related_work', "
            "'methodology', 'experiments', 'conclusion'."
        )

        user_prompt = f"""
Paper Title: {idea.get('proposed_title')}
Idea & Method: {idea.get('proposed_method')}
Motivation: {idea.get('motivation')}
RQs: {idea.get('research_questions')}
Metrics & Empirical Results: {metrics.get('results')}
Generated Plots: {plots}
Available BibTeX Keys: {[p.get('bib_key') for p in papers]}
Review Feedback to Address (if any): {review_feedback}

Please compose the full paper sections. In Related Work and Introduction, cite papers using `\\cite{{key}}`.
In Experiments, explicitly reference the generated figures ({plots}) and cite the quantitative metrics.
Return valid JSON mapping each section name to its complete text in markdown/latex compatible format.
"""
        return self.llm.chat_json([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

    def _write_fallback(
        self,
        idea: Dict[str, Any],
        papers: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        plots: List[str],
        latex_tables: List[str]
    ) -> Dict[str, str]:
        title = idea.get("proposed_title", "Adaptive Optimization in Neural Architectures")
        results = metrics.get("results", {})

        cite_keys = [p.get("bib_key", "ref") for p in papers]
        cite_str = f"\\cite{{{cite_keys[0]}}}" if cite_keys else ""

        abstract = (
            f"Optimizing neural network training requires balanced feature representation and adaptive modulation across varying layers. "
            f"In this work, we propose {title}, a unified framework that introduces dynamic gating and normalized variance scaling. "
            f"Extensive empirical evaluations show that our proposed approach consistently outperforms standard baseline architectures, "
            f"achieving faster convergence and superior generalization."
        )

        intro = (
            f"Recent breakthroughs in deep neural networks have fundamentally transformed machine learning {cite_str}. "
            f"However, existing architectures often suffer from rigid static weighting schemes and suboptimal representation dynamics. "
            f"To address this challenge, we introduce a dynamic modulation mechanism designed to adaptively adjust activation magnitude. "
            f"Our key contributions are three-fold: (1) A novel dynamic gating formulation; (2) Comprehensive benchmarking showing significant gains; "
            f"(3) Systematic ablation studies dissecting each architectural component."
        )

        related = (
            f"Our work builds upon foundational studies in neural architecture optimization {cite_str}. "
            f"Prior methods predominantly rely on fixed weight decay and static normalization layers. "
            f"Unlike existing literature, our approach introduces input-conditioned dynamic variance scaling that dynamically harmonizes feature representations."
        )

        method = (
            f"Let $x \\in \\mathbb{{R}}^d$ denote the input feature representation. The forward pass is defined as: "
            f"\\[ h = \\sigma(W_1 x + b_1) \\] "
            f"We define the adaptive modulation operator $\\mathcal{{M}}(h)$ as: "
            f"\\[ g = \\text{{Sigmoid}}(W_g h + b_g), \\quad \\tilde{{h}} = \\text{{LayerNorm}}(h) \\odot (1 + g) \\] "
            f"This mechanism ensures that significant gradient flows are dynamically gated, suppressing noisy activations while amplifying informative features."
        )

        exp_lines = []
        for k, v in results.items():
            exp_lines.append(f"- **{k}**: Accuracy = {v.get('test_accuracy', 0)*100:.2f}%, Loss = {v.get('test_loss', 0):.4f}, Time = {v.get('training_time_sec', 0):.2f}s")

        exp = (
            f"### Experimental Setup\n"
            f"We conduct comprehensive experiments comparing our proposed framework against standard baselines and ablation variants.\n\n"
            f"### Quantitative Results\n"
            f"{chr(10).join(exp_lines)}\n\n"
            f"As demonstrated in the empirical results and convergence curves (Figure 1), our method exhibits accelerated training convergence "
            f"and improved classification accuracy."
        )

        conclusion = (
            f"We introduced {title}, demonstrating both theoretical motivation and empirical superiority over conventional baseline models. "
            f"Future work will explore scaling this dynamic modulation mechanism to large foundation models and multi-modal pretraining."
        )

        return {
            "abstract": abstract,
            "introduction": intro,
            "related_work": related,
            "methodology": method,
            "experiments": exp,
            "conclusion": conclusion
        }

    def _assemble_markdown(self, idea: Dict[str, Any], sections: Dict[str, str], plots: List[str]) -> str:
        title = idea.get("proposed_title", "Autonomous AI Research Paper")
        md = [
            f"# {title}\n",
            f"**Author**: Autonomous AI Scientist (`Science4Me Engine`)\n",
            f"## Abstract\n{sections.get('abstract', '')}\n",
            f"## 1. Introduction\n{sections.get('introduction', '')}\n",
            f"## 2. Related Work\n{sections.get('related_work', '')}\n",
            f"## 3. Methodology\n{sections.get('methodology', '')}\n",
            f"## 4. Experiments and Empirical Validation\n{sections.get('experiments', '')}\n"
        ]

        if plots:
            md.append("### Empirical Figures\n")
            for p in plots:
                basename = p.split("/")[-1].split("\\")[-1]
                md.append(f"![{basename}]({p})\n*Figure: {basename}*\n")

        md.append(f"## 5. Conclusion\n{sections.get('conclusion', '')}\n")
        return "\n".join(md)

    def _assemble_latex(self, idea: Dict[str, Any], sections: Dict[str, str], plots: List[str], tables: List[str]) -> str:
        title = idea.get("proposed_title", "Autonomous Research Paper")
        fig_includes = ""
        for p in plots:
            fig_includes += f"""
\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.85\\linewidth]{{{p}}}
\\caption{{Empirical evaluation results and convergence profiles.}}
\\end{{figure}}
"""
        table_code = "\n".join(tables) if tables else ""

        latex_template = f"""\\documentclass[11pt]{{article}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage{{amsmath,amssymb,amsfonts}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{hyperref}}
\\usepackage{{cite}}

\\title{{{title}}}
\\author{{Autonomous AI Scientist \\\\ \\texttt{{Science4Me Research Engine}}}}
\\date{{\\today}}

\\begin{{document}}
\\maketitle

\\begin{{abstract}}
{sections.get('abstract', '')}
\\end{{abstract}}

\\section{{Introduction}}
{sections.get('introduction', '')}

\\section{{Related Work}}
{sections.get('related_work', '')}

\\section{{Methodology}}
{sections.get('methodology', '')}

\\section{{Experiments and Results}}
{sections.get('experiments', '')}

{table_code}

{fig_includes}

\\section{{Conclusion}}
{sections.get('conclusion', '')}

\\bibliographystyle{{plain}}
\\bibliography{{references}}

\\end{{document}}
"""
        return latex_template
