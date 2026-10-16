from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


class PaperItem(BaseModel):
    """学术文献元数据对象"""
    title: str = Field(description="论文标题")
    authors: List[str] = Field(default_factory=list, description="作者列表")
    abstract: str = Field(default="", description="摘要")
    arxiv_id: str = Field(default="", description="ArXiv ID")
    published: str = Field(default="", description="发表时间")
    url: str = Field(default="", description="论文链接")
    bibtex: str = Field(default="", description="BibTeX 格式引用条目")


class ResearchState(TypedDict):
    """
    全局科研流状态字典 (符合 LangGraph 状态机标准)
    按照科研生命周期分类排列：
    """
    # 基础信息
    topic: str
    run_id: str
    work_dir: str
    status: str

    # 1. 选题与文献检索 (Ideation)
    papers: List[Dict[str, Any]]
    idea: Dict[str, Any]

    # 2. 实验规划 (Planning)
    experiment_plan: Dict[str, Any]

    # 3. 代码生成与沙箱执行 (Coding & Sandbox)
    code_script: str
    execution_logs: str
    execution_success: bool
    error_traceback: str
    retry_count: int

    # 4. 图表与数据分析 (Visualization & Metrics)
    metrics: Dict[str, Any]
    plots: List[str]
    latex_tables: List[str]

    # 5. 学术论文产物 (Paper Writing)
    paper_sections: Dict[str, str]
    paper_bibtex: str
    paper_markdown: str
    paper_latex: str

    # 6. 同行评审与自我迭代 (Peer Review)
    review: Dict[str, Any]
    revision_count: int
