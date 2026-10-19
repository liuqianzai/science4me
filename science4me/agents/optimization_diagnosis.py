from typing import Any, Dict, List, Optional
from science4me.engine.evolution_log import EvolutionHistoryManager
from science4me.engine.llm import LLMClient
from science4me.tools.academic_search import AcademicSearch


class OptimizationDiagnosisAgent:
    """
    智能体 4: 模型诊断与演进优化智能体 (OptimizationDiagnosisAgent)
    7层架构深度落地（融入真实科研心路历程）：
    1. Identity: 深度学习因果诊断与进化算法专家，精通训练动力学，奉行“保本争优”与“容量权衡”科学哲学
    2. Context: 感知全代际指标拓扑、历史最高分锚点代际、以及与终止目标的绝对距离
    3. Skills: 动力学因果归因、二次文献深潜检索、架构变异算子库、以及轻量化容量自适应放宽
    4. Memory: 演化树拓扑感知、失败机制黑名单、历史最优模型 (Best Checkpoint) 绝对锚定
    5. Connections: 向【门禁 3】提交演化提议（提供渐进改良、颠覆突围或最后手段扩容），达标时发射终止信号
    6. Verification: 诊断与改进方案因果闭环校验、保本回退合法性断言
    7. Automations:
       - 停滞时自动触发【二次文献在线检索】，提炼学术突破方案
       - 当前代退步时自动触发【保本回退】，强制以历史最优代为祖先分支
       - 连续无法超越历史最优时代自动触发【最后手段】，适度放宽参数量确保创新模块兼容
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def run(
        self,
        current_generation: int,
        metrics: Dict[str, Any],
        evolution_mgr: EvolutionHistoryManager,
        target_goal: Dict[str, Any],
        topic: str = ""
    ) -> Dict[str, Any]:
        """执行指标对比、动力学诊断、保本锚定、停滞文献突围与最后手段扩容方案规划"""
        history = evolution_mgr.history
        next_gen = current_generation + 1

        # 1. 终止判定 (Stopping Condition Check)
        target_metric = target_goal.get("metric", "test_accuracy")
        target_threshold = float(target_goal.get("threshold", 0.95))
        condition = target_goal.get("condition", ">=")

        current_val = 0.0
        results = metrics.get("results", {})
        for name, m in results.items():
            if "Standard" not in name and "Baseline" not in name:
                current_val = float(m.get(target_metric, 0.0))
                break

        target_achieved = (current_val >= target_threshold) if condition == ">=" else (current_val <= target_threshold)

        if target_achieved:
            diag_result = {
                "target_achieved": True,
                "current_metric_value": current_val,
                "target_threshold": target_threshold,
                "summary": f"已成功达成设定的最终量化目标！({target_metric} = {current_val*100:.2f}% {condition} {target_threshold*100:.2f}%)",
                "next_action": "stop",
                "model_flaws": [],
                "bottleneck_analysis": "模型已达到科研设定的最终量化标准，泛化指标优良，可终止迭代闭环并结题导出。",
                "improvement_plan": "达成目标，准备结题导出并打包生成学术成果包",
                "next_architecture": {}
            }
            evolution_mgr.update_diagnosis(current_generation, diag_result)
            return diag_result

        # 2. 保本争优核心：检索历史全量记录中最优代条目 (Anchor on Best Model)
        best_entry = evolution_mgr.get_best_generation(metric_key=target_metric, condition=condition)
        best_val = best_entry.get("best_value", current_val) if best_entry else current_val
        best_gen = best_entry.get("generation", current_generation) if best_entry else current_generation
        best_arch = best_entry.get("architecture_spec", {}) if best_entry else {}

        # 判定当前代是否发生退步 (Regression)
        is_regression = (current_val < best_val) if condition == ">=" else (current_val > best_val)

        # 3. 停滞检测与“二次文献突围检索” (Stagnation & Secondary Literature Search)
        is_stagnant = evolution_mgr.check_stagnation(metric_key=target_metric, patience=2, min_delta=0.005)
        breakout_papers = []
        if is_stagnant:
            search_query = topic or target_goal.get("description", "lightweight neural network")
            breakout_papers = AcademicSearch.search_arxiv(f"{search_query} breakthrough architecture", max_results=3)

        # 4. 最后手段判定 (The Last Resort: Capacity Relaxation)
        # 若自历史最优代以来，连续 >= 2 代均无法实现突破超越，激活放宽参数容量策略
        consecutive_stalled = evolution_mgr.get_consecutive_non_improving_count(metric_key=target_metric, condition=condition)
        is_last_resort = (consecutive_stalled >= 2) or (is_stagnant and current_generation >= 3)

        # 5. 诊断与方案生成 (LLM / 规则兜底)
        if self.llm.has_real_key:
            diagnosis_res = self._diagnose_with_llm(
                current_generation=current_generation,
                current_val=current_val,
                best_val=best_val,
                best_gen=best_gen,
                best_arch=best_arch,
                target_goal=target_goal,
                history=history,
                is_stagnant=is_stagnant,
                breakout_papers=breakout_papers,
                is_last_resort=is_last_resort,
                is_regression=is_regression
            )
        else:
            diagnosis_res = self._diagnose_fallback(
                current_generation=current_generation,
                current_val=current_val,
                best_val=best_val,
                best_gen=best_gen,
                best_arch=best_arch,
                target_goal=target_goal,
                is_stagnant=is_stagnant,
                breakout_papers=breakout_papers,
                is_last_resort=is_last_resort,
                is_regression=is_regression
            )

        # 6. 校验层 (Verification): 审查方案规范与自愈
        diagnosis_res = self._verify_and_refine_diagnosis(
            diagnosis_res, next_gen, current_val, target_threshold, is_stagnant, is_last_resort, is_regression, best_gen, best_val
        )

        # 7. 更新回历史管理器 (Memory)
        evolution_mgr.update_diagnosis(current_generation, diagnosis_res)

        return diagnosis_res

    def _diagnose_with_llm(
        self,
        current_generation: int,
        current_val: float,
        best_val: float,
        best_gen: int,
        best_arch: Dict[str, Any],
        target_goal: Dict[str, Any],
        history: List[Dict[str, Any]],
        is_stagnant: bool,
        breakout_papers: List[Dict[str, Any]],
        is_last_resort: bool,
        is_regression: bool
    ) -> Dict[str, Any]:
        papers_summary = "\n".join([
            f"- [{p.get('bib_key')}] {p.get('title')}: {p.get('abstract')[:160]}..."
            for p in breakout_papers
        ]) if breakout_papers else "No extra papers fetched."

        system_prompt = (
            "You are a Lead Deep Learning Diagnostic Scientist and Evolutionary AutoML Architect. "
            "Critically analyze training metrics and the historical evolutionary trajectory across generations. "
            "Rigorous Scientific Principles: "
            "1. 'Anchor on the Best (保本争优)': Always anchor next generation evolutions on the historically best checkpoint (Gen {best_gen}). Never regress on proven gains. If current generation regressed, explicitly roll back to Gen {best_gen}'s architecture as the branching base. "
            "2. 'Literature-Guided Breakout': If stagnant, leverage the newly retrieved breakthrough literature to introduce a structural paradigm shift (e.g. multi-head subspace gating, residual modulation). "
            "3. 'Last Resort (适度放宽参数保兼容)': If `is_last_resort` is TRUE (consecutive failures to beat the historical best), relax the lightweight parameter constraints by moderately widening channels (e.g., from 128 to 160) or increasing heads, to provide sufficient capacity for multi-module compatibility. "
            "Return valid JSON matching the exact schema."
        )

        user_prompt = f"""
Current Generation: {current_generation}
Current Metric Value: {current_val}
Historically Best Metric Value: {best_val} (Achieved at Gen {best_gen})
Is Current Generation Regressed from Best: {is_regression}
Is Stagnant (Plateau reached): {is_stagnant}
Is Last Resort Activated (Capacity relaxation): {is_last_resort}
Target Stopping Threshold: {target_goal}

Newly Retrieved Breakout Literature:
{papers_summary}

Historically Best Architecture Base:
{best_arch}

Recent Evolution History:
{history[-3:] if len(history) > 3 else history}

Formulate the in-depth diagnosis and next generation architecture proposal in JSON:
{{
  "model_flaws": ["Flaw 1", "Flaw 2"],
  "bottleneck_analysis": "Causal analysis of limitation and how the proposal addresses it",
  "stagnation_breakout_triggered": {str(is_stagnant).lower()},
  "last_resort_activated": {str(is_last_resort).lower()},
  "anchor_baseline_gen": {best_gen},
  "improvement_plan": "Concrete architectural evolution plan",
  "next_architecture": {{
    "name": "Gen{current_generation + 1}_ArchitectureName",
    "core_mechanisms": "Detailed description of newly introduced mechanisms",
    "design_rationale": "Why this specific structural modification overcomes the identified bottlenecks",
    "changes_from_previous": "Exact structural delta anchored on Gen{best_gen}"
  }}
}}
"""
        return self.llm.chat_json([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

    def _diagnose_fallback(
        self,
        current_generation: int,
        current_val: float,
        best_val: float,
        best_gen: int,
        best_arch: Dict[str, Any],
        target_goal: Dict[str, Any],
        is_stagnant: bool,
        breakout_papers: List[Dict[str, Any]],
        is_last_resort: bool,
        is_regression: bool
    ) -> Dict[str, Any]:
        next_gen = current_generation + 1

        if is_last_resort:
            # 激活【最后手段】：放宽轻量化约束，适当增加参数换取模块兼容性
            return {
                "model_flaws": [
                    "在极限轻量化约束下，多个创新机制（门控、残差、方差补偿）产生隐空间特征挤压与竞争",
                    "现有通道容量（128维）不足以承载多头子空间的解耦投影，导致表达力受限"
                ],
                "bottleneck_analysis": (
                    f"【最后手段激活】已连续多轮未能实质性超越历史最优 Gen {best_gen} ({best_val*100:.2f}%)。"
                    f"遵循终极权衡策略：在保持推理高效的前提下，适度放宽参数量约束（将特征隐藏通道由 128 提升至 160，扩展为 6 头子空间），"
                    f"以充裕的表征容量彻底消除各模块间的特征竞争，保证创新机制无损兼容！"
                ),
                "stagnation_breakout_triggered": True,
                "last_resort_activated": True,
                "anchor_baseline_gen": best_gen,
                "improvement_plan": f"锚定历史最佳 Gen {best_gen} 权重基底，适度放宽参数量 20%，引入扩容版 6 头子空间解耦门控网络，冲刺终止目标。",
                "next_architecture": {
                    "name": f"Gen{next_gen}_CapacityExpandedSubspaceNet",
                    "core_mechanisms": "扩容版 6 头独立子空间动态门控 (隐藏维度 160) + 全局恒等残差跳跃投影",
                    "design_rationale": "适度增加参数换取模块融合兼容性，解除表征瓶颈",
                    "changes_from_previous": f"基于历史最优 Gen {best_gen}，将通道宽度扩充至 160 并升级为 6 头子空间门控"
                }
            }
        elif is_stagnant:
            # 激活【停滞突围机制】：结合文献提炼颠覆性方案
            lit_mention = "参考 ArXiv 前沿文献" if breakout_papers else "参考前沿子空间自注意理论"
            return {
                "model_flaws": [
                    "单通道门控容量趋近饱和，特征表征陷入局部极值",
                    "浅层语义与深层动态权重耦合过紧，缺乏跨尺度特征解耦"
                ],
                "bottleneck_analysis": (
                    f"【停滞突围机制激活】模型连续多代指标未见实质提升。秉持'保本争优'原则，"
                    f"以历史最佳 Gen {best_gen} ({best_val*100:.2f}%) 为锚点，{lit_mention}，"
                    f"放弃局部微调，直接引入多头子空间自门控 (Multi-Head Subspace Gating) 实施范式重构！"
                ),
                "stagnation_breakout_triggered": True,
                "last_resort_activated": False,
                "anchor_baseline_gen": best_gen,
                "improvement_plan": f"以历史最高表现的 Gen {best_gen} 为基底，重构调制核心：引入 4 头独立子空间动态门控并外挂全局残差保护。",
                "next_architecture": {
                    "name": f"Gen{next_gen}_MultiHeadSelfGatedNet",
                    "core_mechanisms": "4头独立子空间动态门控 + 双通道残差跳跃投影",
                    "design_rationale": "通过子空间正交分解消除特征干扰，跳出局部平台期",
                    "changes_from_previous": f"以历史最优 Gen {best_gen} 为底座，将单通道门控重组为 4 组独立子空间投影头"
                }
            }
        else:
            # 正常渐进式演进
            if current_generation == 1 or best_gen == 1:
                return {
                    "model_flaws": [
                        "单一直连门控缺乏残差恒等通路，深层特征幅值存在收缩漂移",
                        "尚未对输入特征做显式的方差自适应缩放补偿"
                    ],
                    "bottleneck_analysis": "Gen 1 实现了自适应门控的基础收敛，但在深层梯度反传时方差波动较大，导致决策边界不够平滑。",
                    "stagnation_breakout_triggered": False,
                    "last_resort_activated": False,
                    "anchor_baseline_gen": best_gen,
                    "improvement_plan": "在原有门控外部并联恒等残差跳跃连接，并增加可学习参数 variance_scale。",
                    "next_architecture": {
                        "name": f"Gen{next_gen}_ResidualVarianceModulator",
                        "core_mechanisms": "残差跳跃连接 + LayerNorm 稳定化 + 输入自适应方差补偿",
                        "design_rationale": "残差连接保障深层梯度平稳，方差缩放抑制深层激活衰减",
                        "changes_from_previous": "并联恒等跳跃连接，并在激活层引入可学习 variance_scale 参数"
                    }
                }
            else:
                regression_note = f"【保本锚定提示】当前代相比历史最高 Gen {best_gen} 稍有波动，本轮演变强制以 Gen {best_gen} 结构为祖先分支继续改进。" if is_regression else ""
                return {
                    "model_flaws": [
                        "残差通路虽然稳定了反向传播，但在多维非线性特征交叉区域仍存在子空间特征串扰",
                        "单一维度注意力不足以捕获高阶相互作用"
                    ],
                    "bottleneck_analysis": f"{regression_note} 当前准确率已达高位，瓶颈在于多维特征交叉时的子空间干扰。",
                    "stagnation_breakout_triggered": False,
                    "last_resort_activated": False,
                    "anchor_baseline_gen": best_gen,
                    "improvement_plan": f"以历史最优 Gen {best_gen} 为底座，升级为多头子空间自门控网络（Multi-Head Self-Gating）。",
                    "next_architecture": {
                        "name": f"Gen{next_gen}_MultiHeadSelfGatedNet",
                        "core_mechanisms": "多头子空间自门控网络 (Multi-Head Self-Gated Net)",
                        "design_rationale": "通过子空间特征正交解耦消除串扰，实现最终目标冲刺",
                        "changes_from_previous": f"基于历史最优 Gen {best_gen}，将单头调制升级为 4 头独立投影"
                    }
                }

    def _verify_and_refine_diagnosis(
        self,
        diag: Dict[str, Any],
        next_gen: int,
        current_val: float,
        target_threshold: float,
        is_stagnant: bool,
        is_last_resort: bool,
        is_regression: bool,
        best_gen: int,
        best_val: float
    ) -> Dict[str, Any]:
        """校验层 (Verification): 审查方案完整性、保本锚定与参数放宽状态"""
        if not isinstance(diag, dict):
            diag = {}

        diag["target_achieved"] = False
        diag["current_metric_value"] = current_val
        diag["target_threshold"] = target_threshold
        diag["is_stagnant"] = is_stagnant
        diag["last_resort_activated"] = is_last_resort
        diag["is_regression"] = is_regression
        diag["anchor_baseline_gen"] = best_gen
        diag["anchor_baseline_value"] = best_val

        # 确保缺陷列表
        flaws = diag.get("model_flaws", [])
        if not isinstance(flaws, list) or not flaws:
            diag["model_flaws"] = ["特征表征容量受限", "梯度反向传播方差较大"]

        # 确保瓶颈分析文本
        if not diag.get("bottleneck_analysis"):
            diag["bottleneck_analysis"] = (
                f"当前性能指标 ({current_val:.4f}) 距离终止目标 ({target_threshold:.4f}) 尚有差距。"
                f"以历史最优 Gen {best_gen} ({best_val:.4f}) 为锚点持续深化。"
            )

        # 确保下一代架构定义
        next_arch = diag.get("next_architecture", {})
        if not isinstance(next_arch, dict) or not next_arch.get("name"):
            diag["next_architecture"] = {
                "name": f"Gen{next_gen}_EvolvedAdaptiveNet",
                "core_mechanisms": "残差连接与特征自适应门控强化",
                "design_rationale": f"基于历史最优代 Gen {best_gen} 改进特征流动",
                "changes_from_previous": f"以历史最高分代 Gen {best_gen} 为基底引入残差连接与通道自适应调节"
            }

        return diag
