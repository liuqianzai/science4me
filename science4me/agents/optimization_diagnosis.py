from typing import Any, Dict, List
from science4me.engine.llm import LLMClient
from science4me.engine.evolution_log import EvolutionHistoryManager


class OptimizationDiagnosisAgent:
    """
    智能体 4: 模型诊断与演进优化智能体
    职责：
    1. 检测是否达成最终目标（停止条件）
    2. 深度分析当前模型的缺点与性能瓶颈
    3. 检测实验是否陷入停滞；当迟迟无进展时，自动激活“停滞突围机制”提出颠覆性架构方案
    4. 规划下一代架构演进方案并提交人工审核
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def run(
        self,
        current_generation: int,
        metrics: Dict[str, Any],
        evolution_mgr: EvolutionHistoryManager,
        target_goal: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行指标对比、瓶颈诊断与演进方案规划"""
        history = evolution_mgr.history
        latest_entry = evolution_mgr.get_latest_generation() or {}

        # 1. 目标判定
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
            return {
                "target_achieved": True,
                "current_metric_value": current_val,
                "target_threshold": target_threshold,
                "summary": f"已成功达成最终设定目标！({target_metric} = {current_val*100:.2f}% {condition} {target_threshold*100:.2f}%)",
                "next_action": "stop",
                "model_flaws": [],
                "improvement_plan": "达成目标，准备结题导出",
                "next_architecture": {}
            }

        # 2. 检测实验是否陷入停滞 (迟迟无进展)
        is_stagnant = evolution_mgr.check_stagnation(metric_key=target_metric, patience=2, min_delta=0.005)

        # 3. 诊断与方案生成
        if self.llm.has_real_key:
            diagnosis_res = self._diagnose_with_llm(
                current_generation, current_val, target_goal, history, is_stagnant
            )
        else:
            diagnosis_res = self._diagnose_fallback(
                current_generation, current_val, target_goal, is_stagnant
            )

        diagnosis_res["target_achieved"] = False
        diagnosis_res["current_metric_value"] = current_val
        diagnosis_res["target_threshold"] = target_threshold
        diagnosis_res["is_stagnant"] = is_stagnant

        # 更新到历史管理器
        evolution_mgr.update_diagnosis(current_generation, diagnosis_res)

        return diagnosis_res

    def _diagnose_with_llm(
        self,
        generation: int,
        current_val: float,
        target_goal: Dict[str, Any],
        history: List[Dict[str, Any]],
        is_stagnant: bool
    ) -> Dict[str, Any]:
        system_prompt = (
            "You are a Senior Deep Learning Optimization Scientist. "
            "Analyze the training performance and historical evolution logs. "
            "Identify the current architecture's specific flaws and bottlenecks. "
            "If the experiment is STAGNANT (little progress over rounds), activate the 'Stagnation Breakout' "
            "mechanism to propose a radical architectural paradigm shift (e.g. multi-scale residual gating, "
            "subspace projections, novel loss terms). Otherwise, propose an incremental evolution. "
            "Return valid JSON matching the schema."
        )

        user_prompt = f"""
Current Generation: {generation}
Current Metric: {current_val} (Target: {target_goal})
Is Progress Stagnant: {is_stagnant}
History: {history[-3:] if len(history) > 3 else history}

Generate the diagnosis and next evolution proposal in JSON:
{{
  "model_flaws": ["specific flaw 1", "specific flaw 2"],
  "bottleneck_analysis": "deep analysis on why the model is falling short of the target",
  "stagnation_breakout_triggered": {str(is_stagnant).lower()},
  "improvement_plan": "clear technical improvement plan",
  "next_architecture": {{
    "name": "Gen{generation + 1}_ArchitectureName",
    "core_mechanisms": "detailed description of new mechanisms",
    "design_rationale": "why this will resolve the identified flaws and reach the target",
    "changes_from_previous": "exact diff from Gen{generation}"
  }}
}}
"""
        return self.llm.chat_json([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

    def _diagnose_fallback(
        self,
        generation: int,
        current_val: float,
        target_goal: Dict[str, Any],
        is_stagnant: bool
    ) -> Dict[str, Any]:
        next_gen = generation + 1

        if is_stagnant:
            # 激活停滞突围机制：颠覆性重构方案
            return {
                "model_flaws": [
                    "特征表征陷入局部极值，单一线性门控容量饱和",
                    "深层反向传播梯度退化，浅层特征与深层解耦严重"
                ],
                "bottleneck_analysis": "【停滞突围触发】模型在当前拓扑下收敛停滞，单纯增加宽度或调参已无法突破。必须采用多尺度残差跳跃与子空间多头门控重构架构。",
                "stagnation_breakout_triggered": True,
                "improvement_plan": "彻底重构中间调制层：采用 Multi-Head Subspace Gating 与残差双通道映射，强制分离特征流，激活不同子空间的表征潜力。",
                "next_architecture": {
                    "name": f"Gen{next_gen}_MultiHeadSelfGatedNet",
                    "core_mechanisms": "4头独立子空间动态门控 + 双通道残差跳跃投影",
                    "design_rationale": "通过解耦特征维度消除特征重叠干扰，跨越性能平台期",
                    "changes_from_previous": "将单通道门控重构为 4 组独立子空间投影，并引入跨层残差直接通路"
                }
            }
        else:
            if generation == 1:
                return {
                    "model_flaws": [
                        "单一直连门控缺乏残差保护，随着特征深入容易引起特征幅值萎缩",
                        "输入特征与门控之间未考虑方差漂移补偿"
                    ],
                    "bottleneck_analysis": "Gen1 实现了基础自适应调制，但由于缺乏残差通路，深层特征梯度的方差波动较大，限制了分类边界的精细划分。",
                    "stagnation_breakout_triggered": False,
                    "improvement_plan": "引入恒等残差跳跃映射，并加入可学习的输入方差缩放参数，稳定深层梯度流向。",
                    "next_architecture": {
                        "name": f"Gen{next_gen}_ResidualVarianceModulator",
                        "core_mechanisms": "残差跳跃连接 + LayerNorm 稳定化 + 输入自适应方差补偿",
                        "design_rationale": "通过残差通道保护原始语义，同时利用方差缩放防止激活值分布坍缩",
                        "changes_from_previous": "在原有动态门控外部并联恒等跳跃连接，并增加可学习参数 variance_scale"
                    }
                }
            else:
                return {
                    "model_flaws": [
                        "残差通路虽然稳定了梯度，但在复杂特征交叉区域仍存在多特征维度的相互干扰",
                        "尚未利用多子空间并行加权"
                    ],
                    "bottleneck_analysis": "当前模型准确率已达高位，但距离最终目标仍有微弱差距，瓶颈在于特征交叉子空间的表达精细度不足。",
                    "stagnation_breakout_triggered": False,
                    "improvement_plan": "升级为多头子空间门控机制（Multi-Head Gating），将隐层特征拆分为多个独立头分别计算门控权重后重组。",
                    "next_architecture": {
                        "name": f"Gen{next_gen}_MultiHeadSelfGatedNet",
                        "core_mechanisms": "多头子空间自门控网络 (Multi-Head Self-Gating)",
                        "design_rationale": "使网络能够在不同的语义子空间独立加权，达到最终停止目标",
                        "changes_from_previous": "将单头调制扩展为 4 头独立门控，并在输出端使用线性融合层聚合"
                    }
                }
