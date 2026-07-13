"""Client cho dịch vụ ngoài.

vllm.py : wrapper AsyncOpenAI cho vLLM (Qwen3-VL) — completion + streaming hallu
          abort + token budget; singleton dùng chung qua get_shared_vllm_client.
"""

from src.clients.vllm import (
    VLLMClient,
    close_shared_clients,
    get_shared_httpx_client,
    get_shared_vllm_client,
)

__all__ = [
    "VLLMClient",
    "close_shared_clients",
    "get_shared_httpx_client",
    "get_shared_vllm_client",
]
