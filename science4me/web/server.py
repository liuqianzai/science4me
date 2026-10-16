import json
import os
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from science4me.web.session_runner import WebSessionRunner

app = FastAPI(title="Science4Me 科研看板服务", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WORKSPACE_ROOT = "workspaces"
runner = WebSessionRunner(workspace_root=WORKSPACE_ROOT)

# 确保目录存在
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ========== 实体类规范定义 (遵循类引用命名规范与单一标准属性原则) ==========

class StartExperimentRequest(BaseModel):
    """启动科研实验请求体"""
    topic: str = Field(..., description="初始科研选题或核心假说")
    mode: str = Field("interactive", description="运行模式: interactive 或 autonomous")
    target_metric: str = Field("test_accuracy", description="停止量化判定指标")
    target_threshold: float = Field(0.95, description="量化停止阈值")
    target_condition: str = Field(">=", description="判定符号")
    epochs: int = Field(15, description="训练轮次")
    learning_rate: float = Field(0.002, description="初始学习率")
    batch_size: int = Field(64, description="批次大小")
    max_generations: int = Field(5, description="最大演进代数")


class GateDecisionRequest(BaseModel):
    """人机门禁审核决策请求体"""
    gate_id: int = Field(..., description="门禁编号 1, 2 或 3")
    decision: str = Field(..., description="决策操作")
    custom_data: Optional[Dict[str, Any]] = Field(default_factory=dict, description="用户自定义微调参数")


# ========== REST API 接口 ==========

@app.get("/api/experiment/current")
async def get_current_experiment():
    """获取当前正在执行的实验状态快照"""
    state = runner.get_state()
    state["logs"] = runner.logs[-200:]  # 最近 200 条日志
    return state


@app.post("/api/experiment/start")
async def start_experiment(req: StartExperimentRequest):
    """启动科研实验闭环"""
    res = await runner.start(
        topic=req.topic,
        mode=req.mode,
        target_metric=req.target_metric,
        target_threshold=req.target_threshold,
        target_condition=req.target_condition,
        epochs=req.epochs,
        learning_rate=req.learning_rate,
        batch_size=req.batch_size,
        max_generations=req.max_generations
    )
    if not res.get("success"):
        raise HTTPException(status_code=400, detail=res.get("message"))
    return res


@app.post("/api/experiment/gate")
async def submit_gate_decision(req: GateDecisionRequest):
    """提交人机协同门禁审批决定"""
    res = await runner.submit_gate_decision(
        gate_id=req.gate_id,
        decision=req.decision,
        custom_data=req.custom_data
    )
    if not res.get("success"):
        raise HTTPException(status_code=400, detail=res.get("message"))
    return res


@app.post("/api/experiment/stop")
async def stop_experiment():
    """终止当前实验"""
    res = await runner.stop()
    if not res.get("success"):
        raise HTTPException(status_code=400, detail=res.get("message"))
    return res


@app.get("/api/history")
async def get_history_runs():
    """检索工作区全部历史实验记录"""
    if not os.path.exists(WORKSPACE_ROOT):
        return []

    runs = []
    for item in sorted(os.listdir(WORKSPACE_ROOT), reverse=True):
        item_path = os.path.join(WORKSPACE_ROOT, item)
        if os.path.isdir(item_path) and item.startswith("run_"):
            evo_file = os.path.join(item_path, "evolution_history.json")
            metrics_file = os.path.join(item_path, "metrics.json")
            paper_file = os.path.join(item_path, "paper.md")
            bundle_file = os.path.join(item_path, "research_bundle.zip")

            has_evo = os.path.exists(evo_file)
            has_paper = os.path.exists(paper_file)
            has_bundle = os.path.exists(bundle_file)

            gen_count = 0
            best_acc = 0.0
            topic = "未知实验"

            if has_evo:
                try:
                    with open(evo_file, "r", encoding="utf-8") as f:
                        evo_data = json.load(f)
                        gen_count = len(evo_data)
                        if evo_data:
                            # 提取最优指标
                            for g in evo_data:
                                res = g.get("metrics", {}).get("results", {})
                                for m in res.values():
                                    best_acc = max(best_acc, float(m.get("test_accuracy", 0)))
                except Exception:
                    pass

            runs.append({
                "run_id": item,
                "generations": gen_count,
                "best_accuracy": best_acc,
                "has_paper": has_paper,
                "has_bundle": has_bundle,
                "created_at": item.replace("run_", "")
            })

    return runs


@app.get("/api/history/{run_id}")
async def get_history_run_detail(run_id: str):
    """获取单个历史实验的完整结构化产物"""
    run_dir = os.path.join(WORKSPACE_ROOT, run_id)
    if not os.path.exists(run_dir):
        raise HTTPException(status_code=404, detail="未找到该实验记录")

    evo_file = os.path.join(run_dir, "evolution_history.json")
    metrics_file = os.path.join(run_dir, "metrics.json")
    paper_file = os.path.join(run_dir, "paper.md")
    fig_dir = os.path.join(run_dir, "figures")

    evo_data = []
    if os.path.exists(evo_file):
        try:
            with open(evo_file, "r", encoding="utf-8") as f:
                evo_data = json.load(f)
        except Exception:
            pass

    metrics_data = {}
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
        except Exception:
            pass

    paper_content = ""
    if os.path.exists(paper_file):
        try:
            with open(paper_file, "r", encoding="utf-8") as f:
                paper_content = f.read()
        except Exception:
            pass

    figures = []
    if os.path.exists(fig_dir):
        figures = [f for f in os.listdir(fig_dir) if f.endswith((".png", ".jpg", ".svg"))]

    return {
        "run_id": run_id,
        "evolution_history": evo_data,
        "metrics": metrics_data,
        "paper_markdown": paper_content,
        "figures": figures
    }


@app.get("/api/artifacts/{run_id}/file/{file_name:path}")
async def get_artifact_file(run_id: str, file_name: str):
    """获取实验生成的产物文件（图表、论文、压缩包等）"""
    file_path = os.path.join(WORKSPACE_ROOT, run_id, file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(file_path)


# ========== WebSocket 实时日志与状态流 ==========

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    runner.register_websocket(websocket)
    try:
        # 初始握手发送当前快照
        state = runner.get_state()
        state["logs"] = runner.logs[-100:]
        await websocket.send_text(json.dumps({"type": "init", "payload": state}, ensure_ascii=False))

        while True:
            data = await websocket.receive_text()
            # 保持心跳
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        runner.unregister_websocket(websocket)
    except Exception:
        runner.unregister_websocket(websocket)


# ========== 根路由提供一体化单页前端 ==========

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Science4Me Web UI 加载中...</h1>")
