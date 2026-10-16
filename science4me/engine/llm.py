import json
import os
import yaml
from typing import Any, Dict, List, Optional
from openai import OpenAI


class LLMClient:
    """统一模型调用客户端，兼容 OpenAI、DeepSeek、Qwen、Ollama 等各类端点"""

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = self._load_config(config_path)
        llm_cfg = self.config.get("llm", {})

        self.api_key = os.environ.get("OPENAI_API_KEY") or llm_cfg.get("api_key", "")
        self.base_url = os.environ.get("OPENAI_BASE_URL") or llm_cfg.get("base_url", "https://api.openai.com/v1")
        self.model = os.environ.get("OPENAI_MODEL") or llm_cfg.get("model", "gpt-4o")
        self.temperature = float(llm_cfg.get("temperature", 0.7))

        self.has_real_key = bool(self.api_key and len(self.api_key.strip()) > 5)
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url) if self.has_real_key else None

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        json_mode: bool = False
    ) -> str:
        """发送多轮对话请求"""
        temp = temperature if temperature is not None else self.temperature
        if not self.client:
            raise RuntimeError(
                "未配置有效 OPENAI_API_KEY。请在 configs/config.yaml 中填写，或设置系统环境变量 OPENAI_API_KEY。"
            )

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temp,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def chat_json(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """请求并自动解析为 JSON 字典"""
        raw = self.chat(messages, json_mode=True)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # 去除可能包裹的 markdown 代码块
            cleaned = raw.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            return json.loads(cleaned.strip())
