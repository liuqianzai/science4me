import os
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from science4me.engine.session import ResearchSession
from science4me.engine.evolution_log import EvolutionHistoryManager

app = FastAPI(
    title="Science4Me Backend API",
    description="人机协同 AI 科研副驾驶后端服务接口",
    version="0.1.0"
)

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 内存会话管理池
sessions: Dict[str, ResearchSession] = {}


def auto_restore_sessions(workspace_root: str = "workspaces"):
    """服务启动或请求时自动扫描工作区目录，恢复历史 session_state.json 会话"""
    if not os.path.exists(workspace_root):
        return
    for dirname in os.listdir(workspace_root):
        dirpath = os.path.join(workspace_root, dirname)
        state_file = os.path.join(dirpath, "session_state.json")
        if os.path.isdir(dirpath) and os.path.exists(state_file):
            if dirname not in sessions:
                try:
                    s = ResearchSession.resume(dirpath)
                    sessions[s.run_id] = s
                except Exception as ex:
                    print(f"[WARN] Failed to auto-resume {dirpath}: {ex}")


# 初始加载已有会话
auto_restore_sessions()


def get_or_resume_session(session_id: str) -> ResearchSession:
    """优先从内存获取，不存在则尝试从磁盘断点恢复"""
    if session_id in sessions:
        return sessions[session_id]

    # 尝试按 session_id 目录恢复
    dirpath = os.path.abspath(os.path.join("workspaces", session_id))
    if os.path.isdir(dirpath) and os.path.exists(os.path.join(dirpath, "session_state.json")):
        s = ResearchSession.resume(dirpath)
        sessions[s.run_id] = s
        return s

    raise HTTPException(status_code=404, detail=f"科研会话未找到: {session_id}")


# ==================== 请求与响应数据契约 ====================
class CreateSessionRequest(BaseModel):
    topic: str = Field(..., description="初始科研选题或探索方向")


class Gate1ReviewRequest(BaseModel):
    action: str = Field(..., description="决策动作: approve(确认通过), modify(修改目标与架构), regenerate(重新发散)")
    modified_topic: Optional[str] = Field(None, description="修改后的课题名称")
    modified_threshold: Optional[float] = Field(None, description="修改后的停止指标阈值")
    new_prompt: Optional[str] = Field(None, description="重新发散提示词")


class Gate2ReviewRequest(BaseModel):
    action: str = Field(..., description="决策动作: approve(确认通过), modify(调整参数)")
    hyperparameters: Optional[Dict[str, Any]] = Field(None, description="自定义调整的超参数 (epochs, lr 等)")


class Gate3ReviewRequest(BaseModel):
    action: str = Field(..., description="决策动作: approve_next(批准演进), modify_next(修改改进方案), terminate(人工终止)")
    custom_architecture_name: Optional[str] = Field(None, description="自定义下一代架构名称")
    custom_mechanisms: Optional[str] = Field(None, description="自定义下一代核心机制")


# ==================== API 路由实现 ====================
@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "Science4Me Backend"}


@app.get("/api/sessions")
def list_sessions():
    """获取所有活跃及历史已恢复的科研会话列表"""
    auto_restore_sessions()
    items = []
    for sid, s in sessions.items():
        items.append({
            "session_id": sid,
            "topic": s.user_topic,
            "generation": s.generation,
            "status": getattr(s, "status", "active"),
            "work_dir": s.work_dir,
            "created_at": getattr(s, "timestamp", ""),
            "target_goal": s.goal_data.get("target_goal", {})
        })
    return {"sessions": items}


@app.post("/api/sessions/create")
def create_session(req: CreateSessionRequest):
    """创建科研会话并触发智能体 1 (文献检索、停止目标与初始架构设计)"""
    session = ResearchSession(topic=req.topic)
    session_id = session.run_id
    sessions[session_id] = session

    # 运行智能体 1
    goal_data = session.agent1_goal.run(session.user_topic)
    session.goal_data = goal_data
    session.current_architecture = goal_data.get("initial_architecture", {})
    session.status = "awaiting_gate1"
    session.save_state()

    return {
        "session_id": session_id,
        "work_dir": session.work_dir,
        "goal_data": goal_data,
        "current_architecture": session.current_architecture,
        "status": session.status
    }


@app.post("/api/sessions/{session_id}/resume")
def resume_session_endpoint(session_id: str):
    """根据 session_id 显式重载恢复历史会话"""
    session = get_or_resume_session(session_id)
    return {
        "session_id": session.run_id,
        "status": session.status,
        "generation": session.generation,
        "goal_data": session.goal_data,
        "current_plan": session.current_plan,
        "current_architecture": session.current_architecture
    }


@app.post("/api/sessions/{session_id}/gate1_review")
def review_gate1(session_id: str, req: Gate1ReviewRequest):
    """人工审核门禁 1: 审查/微调假说与停止目标，通过后触发智能体 2 生成实验方案"""
    session = get_or_resume_session(session_id)

    if req.action == "modify":
        if req.modified_topic:
            session.goal_data["topic"] = req.modified_topic
        if req.modified_threshold is not None:
            session.goal_data["target_goal"]["threshold"] = float(req.modified_threshold)
    elif req.action == "regenerate":
        if req.new_prompt:
            session.user_topic = req.new_prompt
        goal_data = session.agent1_goal.run(session.user_topic)
        session.goal_data = goal_data
        session.current_architecture = goal_data.get("initial_architecture", {})
        session.status = "awaiting_gate1"
        session.save_state()
        return {
            "session_id": session_id,
            "goal_data": goal_data,
            "current_architecture": session.current_architecture,
            "status": "awaiting_gate1"
        }

    # 门禁 1 通过，触发智能体 2 (方案设计)
    plan = session.agent2_plan.run(
        topic=session.goal_data.get("topic", session.user_topic),
        target_goal=session.goal_data.get("target_goal", {}),
        architecture_spec=session.current_architecture
    )
    session.current_plan = plan
    session.status = "awaiting_gate2"
    session.save_state()

    return {
        "session_id": session_id,
        "plan": plan,
        "status": "awaiting_gate2"
    }


@app.post("/api/sessions/{session_id}/gate2_review")
def review_gate2(session_id: str, req: Gate2ReviewRequest):
    """人工审核门禁 2: 审查实验配置与硬件参数"""
    session = get_or_resume_session(session_id)

    if req.action == "modify" and req.hyperparameters:
        session.current_plan["hyperparameters"].update(req.hyperparameters)

    session.status = "ready_for_training"
    session.save_state()

    return {
        "session_id": session_id,
        "plan": session.current_plan,
        "status": "ready_for_training"
    }


@app.post("/api/sessions/{session_id}/train_step")
def run_train_step(session_id: str):
    """触发智能体 3 (CUDA训练、最佳权重持久化) + 智能体 4 (模型缺陷诊断与演进规划)"""
    session = get_or_resume_session(session_id)

    target_goal = session.goal_data.get("target_goal", {})

    # 1. 智能体 3 训练执行
    success, metrics, logs = session.agent3_train.generate_and_train(
        generation=session.generation,
        architecture_spec=session.current_architecture,
        experiment_plan=session.current_plan,
        work_dir=session.work_dir,
        evolution_mgr=session.evolution_mgr,
        target_goal=target_goal,
        changes_description=session.changes_description
    )

    # 2. 智能体 4 诊断与优化
    diagnosis = session.agent4_diag.run(
        current_generation=session.generation,
        metrics=metrics,
        evolution_mgr=session.evolution_mgr,
        target_goal=target_goal
    )

    target_achieved = diagnosis.get("target_achieved", False)
    session.status = "target_achieved" if target_achieved else "awaiting_gate3"
    session.save_state()

    return {
        "session_id": session_id,
        "generation": session.generation,
        "execution_success": success,
        "metrics": metrics,
        "diagnosis": diagnosis,
        "target_achieved": target_achieved,
        "status": session.status
    }


@app.post("/api/sessions/{session_id}/gate3_review")
def review_gate3(session_id: str, req: Gate3ReviewRequest):
    """人工审核门禁 3: 审查缺陷诊断与下一代演进方案"""
    session = get_or_resume_session(session_id)

    if req.action == "terminate":
        session.status = "terminated_by_user"
        session.save_state()
        return {
            "session_id": session_id,
            "status": "terminated_by_user",
            "generation": session.generation
        }

    latest = session.evolution_mgr.get_latest_generation() or {}
    diagnosis = latest.get("diagnosis", {})
    next_arch = diagnosis.get("next_architecture", {})

    if req.action == "modify_next":
        custom_name = req.custom_architecture_name or next_arch.get("name", f"Gen{session.generation+1}_Custom")
        custom_mech = req.custom_mechanisms or next_arch.get("core_mechanisms", "人工指定定制机制")
        session.current_architecture = {
            "name": custom_name,
            "core_mechanisms": custom_mech,
            "design_rationale": "人工指导微调方案",
            "changes_from_previous": "人工指定架构变动"
        }
        session.changes_description = "人工指导微调方案"
    else:  # approve_next
        session.changes_description = next_arch.get("changes_from_previous", f"由 Gen{session.generation} 演进")
        session.current_architecture = next_arch

    session.generation += 1
    session.status = "ready_for_training"
    session.save_state()

    return {
        "session_id": session_id,
        "next_generation": session.generation,
        "current_architecture": session.current_architecture,
        "status": "ready_for_next_generation"
    }


@app.get("/api/sessions/{session_id}/history")
def get_session_history(session_id: str):
    """获取指定科研会话的完整架构演进谱系记录"""
    session = get_or_resume_session(session_id)

    return {
        "session_id": session_id,
        "history": session.evolution_mgr.history,
        "log_file": session.evolution_mgr.log_file
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
