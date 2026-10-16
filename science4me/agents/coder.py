import re
from typing import Any, Dict, Optional
from science4me.engine.llm import LLMClient


class CoderAgent:
    """代码生成、执行自愈与调试智能体"""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def generate_code(self, idea: Dict[str, Any], plan: Dict[str, Any]) -> str:
        """初次根据实验方案生成可独立运行的 Python 实验代码"""
        if self.llm.has_real_key:
            return self._generate_with_llm(idea, plan)
        else:
            return self._generate_fallback(idea, plan)

    def fix_code(self, previous_code: str, error_traceback: str, plan: Dict[str, Any]) -> str:
        """自愈调试：根据报错栈和上一版代码生成修复后的完整可运行代码"""
        if self.llm.has_real_key:
            return self._fix_with_llm(previous_code, error_traceback, plan)
        else:
            # 基础自愈规则过滤（常见如未引入模块、编码等）
            return self._fix_fallback(previous_code, error_traceback)

    def _generate_with_llm(self, idea: Dict[str, Any], plan: Dict[str, Any]) -> str:
        system_prompt = (
            "You are an expert PyTorch and Machine Learning Software Engineer. "
            "Write a completely self-contained, robust, executable Python script named `experiment.py`. "
            "The script MUST: "
            "1. Train all baselines, proposed model, and ablations specified in the experiment plan. "
            "2. Ensure complete execution within 2 minutes on CPU or GPU. "
            "3. Output all training histories and final evaluation metrics into a JSON file named `metrics.json` in the current working directory. "
            "4. Print progress clearly to stdout. "
            "Output ONLY clean, runnable Python code, without markdown explanation."
        )

        user_prompt = f"""
Research Proposal:
Title: {idea.get('proposed_title')}
Method: {idea.get('proposed_method')}

Experiment Plan:
{plan}

Requirements:
- Must save `metrics.json` with keys: 'results' (dict mapping model_name -> dict of metrics like test_accuracy, test_loss, training_time_sec) and 'history' (dict mapping model_name -> dict with 'loss' and 'val_accuracy' lists).
- Do NOT generate plotting code here (plotting is handled by a downstream agent).
- Make sure code handles exceptions gracefully.
"""
        raw_code = self.llm.chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ], temperature=0.2)
        return self._extract_code(raw_code)

    def _fix_with_llm(self, previous_code: str, error_traceback: str, plan: Dict[str, Any]) -> str:
        system_prompt = (
            "You are a Senior Machine Learning Debugging Specialist. "
            "The previously generated experiment script failed during execution. "
            "Analyze the traceback, locate the root cause (e.g. tensor dimension mismatch, missing import, CUDA/CPU device conflict), "
            "and output the complete corrected, runnable Python code. "
            "Output ONLY clean runnable Python code without markdown chat."
        )

        user_prompt = f"""
Previous Code:
```python
{previous_code}
```

Execution Error Traceback:
{error_traceback}

Please fix the error and return the full revised runnable script.
"""
        raw_code = self.llm.chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ], temperature=0.2)
        return self._extract_code(raw_code)

    def _extract_code(self, text: str) -> str:
        text = text.strip()
        code_match = re.search(r"```python(.*?)```", text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        code_match = re.search(r"```(.*?)```", text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        return text

    def _generate_fallback(self, idea: Dict[str, Any], plan: Dict[str, Any]) -> str:
        return '''import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 设置随机种子保证可复现性
torch.manual_seed(42)

# 1. 准备基准分类数据集
X, y = make_classification(
    n_samples=2500,
    n_features=32,
    n_informative=24,
    n_redundant=4,
    n_classes=3,
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# 2. 定义模型架构
class StandardBaseline(nn.Module):
    """基线 1: 标准全连接多层感知机"""
    def __init__(self, in_features=32, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class StaticModulation(nn.Module):
    """基线 2: 静态加权缩放模型"""
    def __init__(self, in_features=32, num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 128)
        self.static_weights = nn.Parameter(torch.ones(128) * 0.5)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.act = nn.ReLU()
    def forward(self, x):
        h = self.act(self.fc1(x)) * self.static_weights
        h = self.act(self.fc2(h))
        return self.fc3(h)

class ProposedAdaptiveModel(nn.Module):
    """创新模型: 自适应动态门控与特征调制网络"""
    def __init__(self, in_features=32, num_classes=3, use_norm=True, use_dynamic_weight=True):
        super().__init__()
        self.use_norm = use_norm
        self.use_dynamic_weight = use_dynamic_weight
        self.fc1 = nn.Linear(in_features, 128)
        self.norm = nn.LayerNorm(128) if use_norm else nn.Identity()
        self.gate = nn.Sequential(
            nn.Linear(128, 128),
            nn.Sigmoid()
        )
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.act = nn.GELU()
    def forward(self, x):
        h = self.act(self.fc1(x))
        h = self.norm(h)
        if self.use_dynamic_weight:
            g = self.gate(h)
            h = h * (1.0 + g)
        h = self.act(self.fc2(h))
        return self.fc3(h)

models_to_test = {
    "Baseline_StandardMLP": StandardBaseline(),
    "Baseline_StaticWeighting": StaticModulation(),
    "Proposed_AdaptiveModulator": ProposedAdaptiveModel(use_norm=True, use_dynamic_weight=True),
    "Ablation_NoNorm": ProposedAdaptiveModel(use_norm=False, use_dynamic_weight=True),
    "Ablation_NoDynamicWeight": ProposedAdaptiveModel(use_norm=True, use_dynamic_weight=False)
}

criterion = nn.CrossEntropyLoss()
results = {}
history = {}
epochs = 15

for model_name, model in models_to_test.items():
    print(f"--> Training {model_name}...")
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
    start_time = time.time()
    loss_history = []
    acc_history = []

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(bx)
        avg_loss = total_loss / len(train_dataset)
        loss_history.append(round(avg_loss, 4))

        # 验证指标
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for bx, by in test_loader:
                preds = model(bx).argmax(dim=-1)
                correct += (preds == by).sum().item()
                total += len(by)
        val_acc = correct / total
        acc_history.append(round(val_acc, 4))

    elapsed = time.time() - start_time
    test_acc = acc_history[-1]
    test_loss = loss_history[-1]

    results[model_name] = {
        "test_accuracy": test_acc,
        "test_loss": test_loss,
        "training_time_sec": round(elapsed, 2)
    }
    history[model_name] = {
        "loss": loss_history,
        "val_accuracy": acc_history
    }
    print(f"    Finished {model_name} in {elapsed:.2f}s | Test Acc: {test_acc:.4f} | Loss: {test_loss:.4f}")

output_payload = {
    "results": results,
    "history": history
}

with open("metrics.json", "w", encoding="utf-8") as f:
    json.dump(output_payload, f, indent=2)
print("--> metrics.json successfully generated!")
'''

    def _fix_fallback(self, previous_code: str, error_traceback: str) -> str:
        # 兜底回退为稳健版本
        return self._generate_fallback({}, {})
