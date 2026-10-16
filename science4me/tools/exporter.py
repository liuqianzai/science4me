import json
import os
import shutil
import zipfile
from typing import Any, Dict, List


class ResearchExporter:
    """科研产物打包与导出工具"""

    @staticmethod
    def export_all(
        work_dir: str,
        paper_markdown: str,
        paper_latex: str,
        paper_bibtex: str,
        metrics: Dict[str, Any],
        review: Dict[str, Any],
        plots: List[str]
    ) -> str:
        """
        将实验论文、数据指标、图表及 BibTeX 打包为完整标准产物，并生成一键压缩包
        """
        os.makedirs(work_dir, exist_ok=True)

        # 1. 导出 Markdown 论文
        md_path = os.path.join(work_dir, "paper.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(paper_markdown)

        # 2. 导出 LaTeX 论文
        tex_path = os.path.join(work_dir, "paper.tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(paper_latex)

        # 3. 导出 BibTeX 文献库
        bib_path = os.path.join(work_dir, "references.bib")
        with open(bib_path, "w", encoding="utf-8") as f:
            f.write(paper_bibtex)

        # 4. 导出指标与评审 JSON
        metrics_path = os.path.join(work_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        review_path = os.path.join(work_dir, "review.json")
        with open(review_path, "w", encoding="utf-8") as f:
            json.dump(review, f, ensure_ascii=False, indent=2)

        # 5. 打包生成 research_bundle.zip (可直传 Overleaf)
        zip_path = os.path.join(work_dir, "research_bundle.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for fname in ["paper.md", "paper.tex", "references.bib", "metrics.json", "review.json", "experiment.py"]:
                fpath = os.path.join(work_dir, fname)
                if os.path.exists(fpath):
                    zipf.write(fpath, arcname=fname)

            figures_dir = os.path.join(work_dir, "figures")
            if os.path.exists(figures_dir):
                for root, _, files in os.walk(figures_dir):
                    for file in files:
                        full_p = os.path.join(root, file)
                        rel_p = os.path.relpath(full_p, work_dir)
                        zipf.write(full_p, arcname=rel_p)

        return zip_path
