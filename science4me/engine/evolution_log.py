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

    def get_latest_generation(self) -> Optional[Dict[str, Any]]:
        return self.history[-1] if self.history else None

    def save(self):
        os.makedirs(self.work_dir, exist_ok=True)
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
