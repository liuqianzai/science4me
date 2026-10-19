import json
import os
import time
from typing import Any, Dict, List, Optional


class EvolutionHistoryManager:
    """
    模型架构演进记录与谱系管理器
    持久化跟踪每一代模型的架构设计、变更点、硬件指标、评测结果及缺陷诊断
    """

    def __init__(self, work_dir: str):
        self.work_dir = work_dir
        self.log_file = os.path.join(work_dir, "evolution_history.json")
        self.history: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, "r", encoding="utf-8") as f:
                    self.history = json.load(f)
            except Exception:
                self.history = []

    def record_generation(
        self,
        generation: int,
        architecture_name: str,
        architecture_spec: Dict[str, Any],
        changes_from_previous: str,
        hyperparameters: Dict[str, Any],
        metrics: Dict[str, Any],
        target_achieved: bool,
        diagnosis: Optional[Dict[str, Any]] = None,
        checkpoint_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """记录一代模型的完整演进数据，包含模型物理权重路径"""
        entry = {
            "generation": generation,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "architecture_name": architecture_name,
            "architecture_spec": architecture_spec,
            "changes_from_previous": changes_from_previous,
            "hyperparameters": hyperparameters,
            "metrics": metrics,
            "target_achieved": target_achieved,
            "checkpoint_path": checkpoint_path or "",
            "diagnosis": diagnosis or {}
        }
        self.history.append(entry)
        self.save()
        return entry

    def update_diagnosis(self, generation: int, diagnosis: Dict[str, Any]):
        """补充或更新某一代的诊断结果"""
        for item in self.history:
            if item.get("generation") == generation:
                item["diagnosis"] = diagnosis
                break
        self.save()

    def check_stagnation(self, metric_key: str = "test_accuracy", patience: int = 2, min_delta: float = 0.005) -> bool:
        """
        检测实验是否迟迟无进展 (平台期/停滞)：
        若最近 patience 代中指标提升均小于 min_delta，则判定为停滞
        """
        if len(self.history) < patience + 1:
            return False

        recent_entries = self.history[-(patience + 1):]
        values = []
        for e in recent_entries:
            m = e.get("metrics", {}).get(metric_key)
            if m is None:
                # 尝试从 results 字典中提取
                res = e.get("metrics", {}).get("results", {})
                for k, v in res.items():
                    if "Proposed" in k or e.get("architecture_name") in k:
                        m = v.get(metric_key)
                        break
            if m is not None:
                values.append(float(m))

        if len(values) < patience + 1:
            return False

        # 检查提升是否持续低于阈值
        improvements = [values[i] - values[i - 1] for i in range(1, len(values))]
        return all(delta < min_delta for delta in improvements)

    def get_best_generation(
        self,
        metric_key: str = "test_accuracy",
        condition: str = ">="
    ) -> Optional[Dict[str, Any]]:
        """
        保本争优核心：从历史全量演化链中，检索出指标最高（或损失最低）的优胜代条目
        """
        if not self.history:
            return None

        best_entry = None
        best_val = -float("inf") if condition == ">=" else float("inf")

        for entry in self.history:
            val = None
            # 1. 尝试从 results 字典提取针对当前待测架构的指标
            results = entry.get("metrics", {}).get("results", {})
            arch_name = entry.get("architecture_name", "")
            for k, v in results.items():
                if arch_name in k or "Proposed" in k or f"Gen{entry.get('generation')}" in k or "CurrentGen" in k:
                    if metric_key in v:
                        val = float(v[metric_key])
                        break
            # 2. 尝试从根指标提取
            if val is None and metric_key in entry.get("metrics", {}):
                val = float(entry["metrics"][metric_key])

            if val is not None:
                is_better = (val > best_val) if condition == ">=" else (val < best_val)
                if is_better:
                    best_val = val
                    best_entry = entry

        if best_entry is not None:
            # 拷贝一份避免外部篡改，并附带明确的最优数值
            res_entry = dict(best_entry)
            res_entry["best_value"] = best_val
            return res_entry

        return self.history[-1]

    def get_consecutive_non_improving_count(
        self,
        metric_key: str = "test_accuracy",
        condition: str = ">="
    ) -> int:
        """
        计算自历史最高优胜代以来，连续未取得实质超越的代数
        用于程序化判定是否应激活【最后手段：放宽轻量化约束/增加参数容量换兼容】
        """
        best_entry = self.get_best_generation(metric_key=metric_key, condition=condition)
        if not best_entry:
            return 0

        best_gen = best_entry.get("generation", 1)
        latest_gen = self.history[-1].get("generation", 1) if self.history else 1
        return max(0, latest_gen - best_gen)

    def get_latest_generation(self) -> Optional[Dict[str, Any]]:
        return self.history[-1] if self.history else None

    def save(self):
        os.makedirs(self.work_dir, exist_ok=True)
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
