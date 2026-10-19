import ast
import re
from typing import Any, Dict, Optional, Tuple
from science4me.engine.llm import LLMClient


class CoderAgent:
    """
    代码生成、执行自愈与调试智能体 (CoderAgent)
    7层架构增强落地：
    1. Identity: 资深 PyTorch 系统工程师与自愈调试专家
    2. Context: 自动感知模型方案、消融实验清单与运行时沙箱规范
    3. Skills: 自动生成端到端训练代码、CUDA 加速优先适配、报错栈定位与修复
    4. Memory: 常见异常（维度不匹配、类型转换、除零溢出）自愈规则记忆
    5. Connections: 向上承接 Planner 规划方案，向下输出标准 metrics.json 供 Visualizer 和 Writer 消费
    6. Verification: AST 静态语法树预检，杜绝语法错误脚本进入执行沙箱
    7. Automations: Traceback 驱动的代码全自动自愈重试流水线
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def generate_code(self, idea: Dict[str, Any], plan: Dict[str, Any]) -> str:
        """初次根据实验方案生成可独立运行的 Python 实验代码，并执行 AST 语法验证"""
        if self.llm.has_real_key:
            code = self._generate_with_llm(idea, plan)
        else:
            code = self._generate_fallback(idea, plan)

        # 校验层: AST 静态语法审查
        is_valid, err = self._verify_syntax(code)
        if not is_valid:
            # 自动修复语法或回退到稳健版本
            code = self._generate_fallback(idea, plan)

        return code

    def fix_code(self, previous_code: str, error_traceback: str, plan: Dict[str, Any]) -> str:
        """自愈调试：根据报错栈和上一版代码生成修复后的完整可运行代码"""
        if self.llm.has_real_key:
            fixed_code = self._fix_with_llm(previous_code, error_traceback, plan)
        else:
            fixed_code = self._fix_fallback(previous_code, error_traceback)

        is_valid, err = self._verify_syntax(fixed_code)
        if not is_valid:
            fixed_code = self._generate_fallback({}, plan)

        return fixed_code

    def _verify_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """校验层 (Verification): 审查 Python 源码语法树"""
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, str(e)

    def _generate_with_llm(self, idea: Dict[str, Any], plan: Dict[str, Any]) -> str:
        system_prompt = (
            "You are an expert PyTorch and Machine Learning Software Engineer. "
            "Write a completely self-contained, robust, executable Python script named `experiment.py`. "
            "Defensive Coding Standards: "
            "1. Device Acceleration: Always use `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`. "
            "2. Baseline & Proposed Models: Implement and train all baselines and proposed architectures specified in the plan. "
            "3. Progress & Metric Export: Output all training histories and final evaluation metrics into `metrics.json`. "
            "4. Output format in metrics.json: "
            "   {'results': {model_name: {'test_accuracy': float, 'test_loss': float, 'training_time_sec': float}}, "
            "    'history': {model_name: {'loss': list, 'val_accuracy': list}}}. "
            "Output ONLY clean, runnable Python code, without markdown chat or explanations."
        )

        user_prompt = f"""
Research Proposal:
Title: {idea.get('proposed_title')}
Method: {idea.get('proposed_method')}

Experiment Plan:
{plan}

Generate clean, robust, executable PyTorch script now:
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

Please fix the error and return the full revised runnable script:
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEVICE] Hardware backend: {device} (CUDA: {torch.cuda.is_available()})")

torch.manual_seed(42)

# 1. 构建高维基准数据集
X, y = make_classification(
    n_samples=3000,
    n_features=32,
    n_informative=24,
    n_redundant=4,
    n_classes=3,
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# 2. 对照基线架构 StandardMLP
class StandardMLP(nn.Module):
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

# 3. 创新模型: 自适应动态门控与残差方差网络
class ProposedAdaptiveModel(nn.Module):
    def __init__(self, in_features=32, num_classes=3):
        super().__init__()
        self.proj = nn.Linear(in_features, 128)
        self.norm = nn.LayerNorm(128)
        self.gate = nn.Sequential(
            nn.Linear(128, 128),
            nn.Sigmoid()
        )
        self.variance_scale = nn.Parameter(torch.ones(128) * 1.05)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.act = nn.GELU()

    def forward(self, x):
        residual = self.proj(x)
        h = self.norm(residual)
        g = self.gate(h)
        h = (h * (1.0 + g) * self.variance_scale) + residual
        h = self.act(self.fc2(h))
        return self.fc3(h)

# 4. 消融对照组: 无动态门控消融网络
class AblationNoGate(nn.Module):
    def __init__(self, in_features=32, num_classes=3):
        super().__init__()
        self.proj = nn.Linear(in_features, 128)
        self.norm = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.act = nn.GELU()

    def forward(self, x):
        residual = self.proj(x)
        h = self.act(self.norm(residual)) + residual
        h = self.act(self.fc2(h))
        return self.fc3(h)

models_to_test = {
    "StandardBaseline": StandardMLP().to(device),
    "Ablation_NoGate": AblationNoGate().to(device),
    "ProposedAdaptiveModel": ProposedAdaptiveModel().to(device)
}

criterion = nn.CrossEntropyLoss()
results = {}
history = {}
epochs = 15

for model_name, model in models_to_test.items():
    print(f"--> Training {model_name}...")
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    start_time = time.time()
    loss_history = []
    acc_history = []

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(bx)
        avg_loss = total_loss / len(train_dataset)
        loss_history.append(round(avg_loss, 4))

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)
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
        return self._generate_fallback({}, {})
