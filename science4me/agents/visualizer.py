import os
from typing import Any, Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")  # 强制无 GUI 后端，确保命令行沙箱环境下稳定生成
import matplotlib.pyplot as plt
import seaborn as sns


class VisualizerAgent:
    """学术级图表与 LaTeX 统计表格生成智能体"""

    def __init__(self):
        # 设定出版级科研绘图样式
        sns.set_theme(style="whitegrid", palette="deep")
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
        plt.rcParams["axes.edgecolor"] = "#333333"
        plt.rcParams["axes.linewidth"] = 0.8

    def run(self, metrics: Dict[str, Any], work_dir: str) -> Tuple[List[str], List[str]]:
        """
        基于实验指标生成学术图表与 LaTeX 汇总表格
        返回: (生成图片路径列表, LaTeX 表格代码列表)
        """
        figures_dir = os.path.join(work_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)

        results = metrics.get("results", {})
        history = metrics.get("history", {})

        generated_plots = []

        # 1. 绘制训练收敛曲线 (Loss Curves)
        loss_plot_path = os.path.join(figures_dir, "loss_curves.png")
        if history:
            plt.figure(figsize=(7, 4.5), dpi=300)
            for model_name, data in history.items():
                loss_vals = data.get("loss", [])
                if loss_vals:
                    plt.plot(range(1, len(loss_vals) + 1), loss_vals, label=model_name, linewidth=2.0)
            plt.xlabel("Epoch", fontsize=11, fontweight="bold")
            plt.ylabel("Training Loss", fontsize=11, fontweight="bold")
            plt.title("Convergence Comparison Across Models", fontsize=12, fontweight="bold", pad=10)
            plt.legend(frameon=True, fontsize=9)
            plt.tight_layout()
            plt.savefig(loss_plot_path, dpi=300)
            plt.close()
            generated_plots.append(os.path.relpath(loss_plot_path, work_dir).replace("\\", "/"))

        # 2. 绘制基线对比柱状图 (Accuracy Comparison)
        bar_plot_path = os.path.join(figures_dir, "benchmark_comparison.png")
        if results:
            plt.figure(figsize=(8, 4.5), dpi=300)
            names = list(results.keys())
            accs = [results[k].get("test_accuracy", 0.0) * 100 for k in names]
            colors = ["#4C72B0" if "Proposed" not in n else "#55A868" for n in names]

            bars = plt.bar(names, accs, color=colors, width=0.55, edgecolor="#222222", linewidth=0.8)
            plt.ylabel("Test Accuracy (%)", fontsize=11, fontweight="bold")
            plt.title("Overall Empirical Performance Benchmarking", fontsize=12, fontweight="bold", pad=10)
            plt.xticks(rotation=20, ha="right", fontsize=9)
            plt.ylim(min(accs) * 0.85 if accs else 0, 102)

            for bar in bars:
                h = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2.0, h + 0.8, f"{h:.2f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

            plt.tight_layout()
            plt.savefig(bar_plot_path, dpi=300)
            plt.close()
            generated_plots.append(os.path.relpath(bar_plot_path, work_dir).replace("\\", "/"))

        # 3. 自动生成 LaTeX 统计对比表格
        latex_tables = []
        if results:
            best_acc = max([v.get("test_accuracy", 0) for v in results.values()]) if results else 0
            best_loss = min([v.get("test_loss", 999) for v in results.values()]) if results else 0

            rows = []
            for name, m in results.items():
                acc = m.get("test_accuracy", 0.0) * 100
                loss = m.get("test_loss", 0.0)
                time_s = m.get("training_time_sec", 0.0)

                acc_str = f"\\textbf{{{acc:.2f}\\%}}" if m.get("test_accuracy", 0) == best_acc else f"{acc:.2f}\\%"
                loss_str = f"\\textbf{{{loss:.4f}}}" if m.get("test_loss", 999) == best_loss else f"{loss:.4f}"

                clean_name = name.replace("_", " ")
                rows.append(f"{clean_name} & {acc_str} & {loss_str} & {time_s:.2f}s \\\\")

            table_code = f"""\\begin{{table}}[t]
\\centering
\\caption{{Quantitative comparison of proposed model against baselines and ablations.}}
\\label{{tab:main_results}}
\\begin{{tabular}}{{lccc}}
\\toprule
\\textbf{{Model Architecture}} & \\textbf{{Test Accuracy}} & \\textbf{{Test Loss}} & \\textbf{{Training Time}} \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
            latex_tables.append(table_code)

        return generated_plots, latex_tables
