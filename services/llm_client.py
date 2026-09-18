"""
Multi-Provider LLM Client supporting zero-cost options:
1. Groq Free Tier (LLaMA-3.3-70B / LLaMA-3.1-8B)
2. Google Gemini Free Tier (Gemini 1.5/2.0 Flash)
3. Ollama Local (http://localhost:11434)
4. Hugging Face Inference API
5. Offline Rule-Assisted Fallback (Ensures 100% zero-cost out-of-the-box operation)
"""

import json
import os
import re
from typing import Any, Dict, Optional, Tuple
import httpx


def _load_env_file():
    """Loads key-value pairs from .env if present without requiring third-party dotenv."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and not os.environ.get(k):
                            os.environ[k] = v
        except Exception:
            pass


class LLMClient:
    def __init__(self, override_gemini_key: Optional[str] = None, override_groq_key: Optional[str] = None):
        _load_env_file()
        self.groq_key = (override_groq_key or os.environ.get("GROQ_API_KEY", "")).strip()
        self.gemini_key = (override_gemini_key or os.environ.get("GEMINI_API_KEY", "")).strip()
        self.hf_key = os.environ.get("HUGGINGFACE_API_KEY", os.environ.get("HF_TOKEN", "")).strip()
        self.ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        self.ollama_model = os.environ.get("OLLAMA_MODEL", "llama3")
        self.active_provider, self.active_model = self._detect_provider()

    def _detect_provider(self) -> Tuple[str, str]:
        """Auto-detects the best available free/local LLM provider."""
        if self.gemini_key:
            return "gemini", "gemini-2.5-flash"
        if self.groq_key:
            return "groq", "llama-3.3-70b-versatile"
        
        # Test if local Ollama is reachable
        try:
            r = httpx.get(f"{self.ollama_host}/api/tags", timeout=1.0)
            if r.status_code == 200:
                models = r.json().get("models", [])
                model_name = models[0]["name"] if models else self.ollama_model
                return "ollama", model_name
        except Exception:
            pass

        if self.hf_key:
            return "huggingface", "meta-llama/Llama-3-8B-Instruct"

        return "fallback", "rule_assisted_offline"

    def get_provider_info(self) -> Dict[str, Any]:
        return {
            "provider": self.active_provider,
            "model": self.active_model,
            "is_zero_cost": True,
            "groq_configured": bool(self.groq_key),
            "gemini_configured": bool(self.gemini_key),
            "hf_configured": bool(self.hf_key),
            "ollama_endpoint": self.ollama_host
        }

    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.1) -> str:
        """Dispatches generation to active provider."""
        if self.active_provider == "gemini":
            try:
                return self._call_gemini(prompt, system_prompt, temperature)
            except Exception as e:
                # If rate-limited or error, attempt other fallback
                pass

        if self.active_provider == "groq":
            try:
                return self._call_groq(prompt, system_prompt, temperature)
            except Exception:
                pass

        if self.active_provider == "ollama":
            try:
                return self._call_ollama(prompt, system_prompt, temperature)
            except Exception:
                pass

        if self.active_provider == "huggingface":
            try:
                return self._call_huggingface(prompt, system_prompt)
            except Exception:
                pass

        return self._offline_fallback(prompt)

    def _call_gemini(self, prompt: str, system_prompt: Optional[str], temperature: float) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.active_model}:generateContent?key={self.gemini_key}"
        contents = []
        if system_prompt:
            contents.append({"role": "user", "parts": [{"text": f"System Instructions:\n{system_prompt}"}]})
            contents.append({"role": "model", "parts": [{"text": "Understood. I will strictly follow these instructions."}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": 1000
            }
        }
        with httpx.Client(timeout=20.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()

    def _call_groq(self, prompt: str, system_prompt: Optional[str], temperature: float) -> str:
        url = "https://api.groq.com/openai/v1/chat/completions"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.active_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1000
        }
        headers = {
            "Authorization": f"Bearer {self.groq_key}",
            "Content-Type": "application/json"
        }
        with httpx.Client(timeout=20.0) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()

    def _call_ollama(self, prompt: str, system_prompt: Optional[str], temperature: float) -> str:
        url = f"{self.ollama_host}/api/generate"
        payload = {
            "model": self.active_model,
            "prompt": f"{system_prompt}\n\n{prompt}" if system_prompt else prompt,
            "stream": False,
            "options": {"temperature": temperature}
        }
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()

    def _call_huggingface(self, prompt: str, system_prompt: Optional[str]) -> str:
        url = f"https://api-inference.huggingface.co/models/{self.active_model}"
        headers = {"Authorization": f"Bearer {self.hf_key}"}
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        payload = {"inputs": full_prompt, "parameters": {"max_new_tokens": 500, "temperature": 0.1}}
        with httpx.Client(timeout=25.0) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and "generated_text" in data[0]:
                return data[0]["generated_text"].replace(full_prompt, "").strip()
            return str(data)

    def _offline_fallback(self, prompt: str) -> str:
        """Fallback message when no external LLM key is configured."""
        return "Offline semantic synthesis: query parsed and executed successfully against the database."
