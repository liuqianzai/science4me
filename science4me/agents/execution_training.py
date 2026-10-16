import json
import os
import re
from typing import Any, Dict, Tuple
from science4me.engine.llm import LLMClient
from science4me.engine.evolution_log import EvolutionHistoryManager
from science4me.tools.sandbox import SandboxExecutor


class ExecutionTrainingAgent:
    """
    智能体 3: 代码执行与训练智能体
    职责：
    1. 编写模型训练与评测代码 (支持 CUDA 硬件加速)
    2. 沙箱执行真实训练，捕获硬件与指标数据
    3. 持久化保存训练结果及架构演进记录 (Evolution Log)
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def generate_and_train(
        self,
        generation: int,
        architecture_spec: Dict[str, Any],
        experiment_plan: Dict[str, Any],
        work_dir: str,
        evolution_mgr: EvolutionHistoryManager,
        target_goal: Dict[str, Any],
        changes_description: str = "初始代架构",
        log_callback=None
    ) -> Tuple[bool, Dict[str, Any], str]:
        """生成训练脚本、在沙箱中执行训练并持久化演进记录"""
        code = self._generate_training_code(generation, architecture_spec, experiment_plan)

        script_name = f"train_gen_{generation}.py"
        script_path = os.path.join(work_dir, script_name)
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)

        # 在沙箱中执行
        if log_callback:
            success, logs, metrics, error_tb = SandboxExecutor.run_script_stream(
                code=code,
                work_dir=work_dir,
                script_name=script_name,
                timeout_seconds=300,
                log_callback=log_callback
            )
        else:
            success, logs, metrics, error_tb = SandboxExecutor.run_script(
                code=code,
                work_dir=work_dir,
                script_name=script_name,
                timeout_seconds=300
            )

        # 检查是否达成目标
        target_metric = target_goal.get("metric", "test_accuracy")
        target_val = float(target_goal.get("threshold", 0.95))
        condition = target_goal.get("condition", ">=")

        current_val = 0.0
        # 从当前模型指标中提取
        arch_name = architecture_spec.get("name", f"Model_Gen{generation}")
        results = metrics.get("results", {})
        for name, m in results.items():
            if arch_name in name or "Proposed" in name or f"Gen{generation}" in name:
                current_val = float(m.get(target_metric, 0.0))
                break

        achieved = (current_val >= target_val) if condition == ">=" else (current_val <= target_val)

        # 提取模型物理权重路径
        checkpoint_path = ""
        for name, m in results.items():
            if arch_name in name or "Proposed" in name or f"Gen{generation}" in name:
                checkpoint_path = m.get("checkpoint_path", "")
                break

        # 持久化记录到演进历史文件
        evolution_mgr.record_generation(
            generation=generation,
            architecture_name=arch_name,
            architecture_spec=architecture_spec,
            changes_from_previous=changes_description,
            hyperparameters=experiment_plan.get("hyperparameters", {}),
            metrics=metrics,
            target_achieved=achieved,
            checkpoint_path=checkpoint_path
        )

        return success, metrics, logs

    def _generate_training_code(
        self,
        generation: int,
        architecture_spec: Dict[str, Any],
        plan: Dict[str, Any]
    ) -> str:
        """根据架构代际生成对应的 PyTorch 训练代码（支持 CUDA）"""
        if self.llm.has_real_key:
            return self._generate_with_llm(generation, architecture_spec, plan)
        else:
            return self._generate_fallback(generation, architecture_spec, plan)

    def _generate_with_llm(
        self,
        generation: int,
        architecture_spec: Dict[str, Any],
        plan: Dict[str, Any]
    ) -> str:
        system_prompt = (
            "You are an expert PyTorch Engineer. Generate a self-contained, executable script. "
            "IMPORTANT: Explicitly check for and utilize CUDA GPU acceleration if available: "
            "`device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`. "
            "Output training metrics into `metrics.json` with 'results' mapping model names to metrics. "
            "Return ONLY executable Python code."
        )

        user_prompt = f"""
Generation: {generation}
Architecture Specification:
{architecture_spec}

Experiment Plan:
{plan}

Requirements:
- Must move models and tensors to `device`.
- Train baselines and the generation model.
- Save `metrics.json`.
"""
        raw_code = self.llm.chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ], temperature=0.2)

        code_match = re.search(r"```python(.*?)```", raw_code, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        code_match = re.search(r"```(.*?)```", raw_code, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        return raw_code.strip()

    def _generate_fallback(
        self,
        generation: int,
        architecture_spec: Dict[str, Any],
        plan: Dict[str, Any]
    ) -> str:
        hyp = plan.get("hyperparameters", {})
        epochs = hyp.get("epochs", 15)
        lr = hyp.get("learning_rate", 0.002)
        arch_name = architecture_spec.get("name", f"Gen{generation}_Model")

        # 根据不同代际装配不同复杂度和能力的网络结构
        if generation == 1:
            gen_model_code = """
class CurrentGenModel(nn.Module):
    \"\"\"Gen 1: 全连接 + LayerNorm + Sigmoid 动态特征门控\"\"\"
    def __init__(self, in_features=32, num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 128)
        self.norm = nn.LayerNorm(128)
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
        g = self.gate(h)
        h = h * (1.0 + g)
        h = self.act(self.fc2(h))
        return self.fc3(h)
"""
        elif generation == 2:
            gen_model_code = """
class CurrentGenModel(nn.Module):
    \"\"\"Gen 2: 残差跳跃连接 + 输入自适应方差补偿调制\"\"\"
    def __init__(self, in_features=32, num_classes=3):
        super().__init__()
        self.proj = nn.Linear(in_features, 128)
        self.norm1 = nn.LayerNorm(128)
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
        h = self.norm1(residual)
        g = self.gate(h)
        h = (h * (1.0 + g) * self.variance_scale) + residual
        h = self.act(self.fc2(h))
        return self.fc3(h)
"""
        else:
            gen_model_code = """
class CurrentGenModel(nn.Module):
    \"\"\"Gen 3+: 多头子空间自门控网络 (Multi-Head Subspace Self-Gating)\"\"\"
    def __init__(self, in_features=32, num_classes=3, heads=4):
        super().__init__()
        self.in_proj = nn.Linear(in_features, 128)
        self.norm = nn.LayerNorm(128)
        self.heads = heads
        self.head_dim = 128 // heads
        self.gate_heads = nn.Linear(128, 128)
        self.mix = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.act = nn.GELU()

    def forward(self, x):
        res = self.in_proj(x)
        h = self.norm(res)
        gates = torch.sigmoid(self.gate_heads(h))
        gated_h = self.mix(h * (1.0 + gates)) + res
        out = self.act(self.fc2(gated_h))
        return self.fc3(out)
"""

        return f'''import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 硬件设备自动识别与 CUDA 硬件加速
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEVICE] Hardware backend: {{device}} (CUDA Available: {{torch.cuda.is_available()}})")

torch.manual_seed(42)

# 数据集构建
X, y = make_classification(
    n_samples=3000,
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

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
    batch_size=64,
    shuffle=True
)
test_loader = DataLoader(
    TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)),
    batch_size=128,
    shuffle=False
)

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

{gen_model_code}

models = {{
    "Baseline_StandardMLP": StandardMLP().to(device),
    "{arch_name}": CurrentGenModel().to(device)
}}

os.makedirs("checkpoints", exist_ok=True)
criterion = nn.CrossEntropyLoss()
results = {{}}
history = {{}}
epochs = {epochs}

for name, model in models.items():
    optimizer = optim.Adam(model.parameters(), lr={lr}, weight_decay=1e-4)
    start_time = time.time()
    loss_hist, acc_hist = [], []
    best_acc = -1.0
    best_state = None

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(bx)
        avg_loss = total_loss / len(X_train)
        loss_hist.append(round(avg_loss, 4))

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)
                preds = model(bx).argmax(dim=-1)
                correct += (preds == by).sum().item()
                total += len(by)
        cur_acc = round(correct / total, 4)
        acc_hist.append(cur_acc)

        if cur_acc > best_acc:
            best_acc = cur_acc
            best_state = {{k: v.cpu().clone() for k, v in model.state_dict().items()}}

    elapsed = time.time() - start_time
    ckpt_path = f"checkpoints/best_{{name}}.pt"
    if best_state is not None:
        torch.save(best_state, ckpt_path)

    results[name] = {{
        "test_accuracy": acc_hist[-1],
        "test_loss": loss_hist[-1],
        "training_time_sec": round(elapsed, 2),
        "checkpoint_path": ckpt_path
    }}
    history[name] = {{
        "loss": loss_hist,
        "val_accuracy": acc_hist
    }}
    print(f"[EVAL] {{name}}: Acc = {{acc_hist[-1]*100:.2f}}% | Loss = {{loss_hist[-1]:.4f}} | Saved Checkpoint: {{ckpt_path}}")

with open("metrics.json", "w", encoding="utf-8") as f:
    json.dump({{"results": results, "history": history}}, f, indent=2)
'''
