"""
Minimal chat-completions helper used by the GRID research artifact.

The goal is to expose only two small public interfaces:

- ask(): one prompt -> one response
- asks(): N prompts -> N responses

The goal is to keep the repository-facing interface compact and portable.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Union

from openai import OpenAI


Message = Dict[str, str]
Prompt = Union[str, Sequence[Message]]
__all__ = ["ask", "asks"]

OPENAI_OFFICIAL_BASE_URL = "https://api.openai.com/v1"
GOOGLE_OPENAI_LIKE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"


def _resolve_messages(prompt: Prompt) -> List[Message]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return [{"role": item["role"], "content": item["content"]} for item in prompt]


def _resolve_model(model: Optional[str]) -> str:
    resolved = model or os.environ.get("OPENAI_MODEL")
    if not resolved:
        raise ValueError("No model provided. Set `model=` or `OPENAI_MODEL`.")
    return resolved


def _make_client(
    model: str,
    api_key: Optional[str],
    base_url: Optional[str],
    timeout: Union[int, float],
) -> OpenAI:
    resolved_model = str(model or "").strip().lower()

    if base_url:
        resolved_base_url = base_url
        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
    elif resolved_model.startswith(("gpt", "o1", "o3", "o4")):
        resolved_base_url = OPENAI_OFFICIAL_BASE_URL
        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
    elif "gemini" in resolved_model:
        resolved_base_url = GOOGLE_OPENAI_LIKE_BASE_URL
        resolved_api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
    else:
        resolved_base_url = os.environ.get("OPENAI_BASE_URL")
        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")

    if not resolved_api_key:
        raise ValueError("No API key provided. Set `api_key=` or the matching environment variable.")
    return OpenAI(
        api_key=resolved_api_key,
        base_url=resolved_base_url,
        timeout=timeout,
        max_retries=0,
    )


def _response_text(response: Any) -> str:
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(str(item.get("text", "")))
            else:
                chunks.append(str(item))
        return "".join(chunks)
    return "" if content is None else str(content)


def ask(
    prompt: Prompt,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_tokens: int = 32768,
    temperature: float = 0.0,
    timeout: Union[int, float] = 300,
    retries: int = 2,
    **completion_kwargs: Any,
) -> str:
    """
    Send a single prompt to a chat-completions endpoint.
    """

    messages = _resolve_messages(prompt)
    resolved_model = _resolve_model(model)
    client = _make_client(
        model=resolved_model,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
    )

    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=resolved_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **completion_kwargs,
            )
            return _response_text(response)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(min(2.0 * (attempt + 1), 5.0))

    raise RuntimeError(f"Chat-completions request failed: {last_error}") from last_error


def asks(
    prompt_list: Sequence[Prompt],
    model: Union[str, Sequence[str], None] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_tokens: int = 32768,
    temperature: float = 0.0,
    concurrency: int = 8,
    timeout: Union[int, float] = 300,
    retries: int = 2,
    **completion_kwargs: Any,
) -> List[str]:
    """
    Send N prompts through N independent chat-completions calls.
    """

    if not prompt_list:
        return []

    if isinstance(model, Sequence) and not isinstance(model, str):
        if len(model) != len(prompt_list):
            raise ValueError("When `model` is a list, it must match `prompt_list` length.")
        model_list = list(model)
    else:
        shared_model = _resolve_model(model if isinstance(model, str) else None)
        model_list = [shared_model] * len(prompt_list)

    results: List[str] = [""] * len(prompt_list)
    with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as executor:
        future_map = {
            executor.submit(
                ask,
                prompt=prompt,
                model=model_list[index],
                api_key=api_key,
                base_url=base_url,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                retries=retries,
                **completion_kwargs,
            ): index
            for index, prompt in enumerate(prompt_list)
        }
        for future in as_completed(future_map):
            index = future_map[future]
            results[index] = future.result()
    return results
