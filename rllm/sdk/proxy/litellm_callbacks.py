"""LiteLLM callbacks for parameter injection and tracing."""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import ModelResponse, ModelResponseStream

from rllm.sdk.tracers import SqliteTracer

logger = logging.getLogger(__name__)


def _extract_rllm_metadata(data: dict[str, Any]) -> dict[str, Any]:
    """Best-effort extraction of RLLM metadata across LiteLLM payload shapes."""

    def _candidate_paths() -> list[Any]:
        metadata = data.get("metadata")
        requester_metadata = data.get("requester_metadata")
        litellm_params = data.get("litellm_params") or {}
        litellm_metadata = litellm_params.get("metadata")
        standard_logging_object = data.get("standard_logging_object") or {}
        standard_metadata = standard_logging_object.get("metadata")
        proxy_request = data.get("proxy_server_request") or {}
        proxy_body = proxy_request.get("body") or {}
        proxy_metadata = proxy_body.get("metadata")
        proxy_requester_metadata = proxy_body.get("requester_metadata")
        return [
            metadata,
            requester_metadata,
            litellm_metadata,
            standard_metadata,
            data.get("rllm_metadata"),
            proxy_metadata,
            proxy_requester_metadata,
        ]

    for candidate in _candidate_paths():
        if not isinstance(candidate, dict):
            continue

        nested = candidate.get("rllm_metadata")
        if isinstance(nested, dict):
            return dict(nested)

        if "session_uids" in candidate or "session_name" in candidate:
            return dict(candidate)

        nested_requester = candidate.get("requester_metadata")
        if isinstance(nested_requester, dict):
            nested = nested_requester.get("rllm_metadata")
            if isinstance(nested, dict):
                return dict(nested)
            if "session_uids" in nested_requester or "session_name" in nested_requester:
                return dict(nested_requester)

    return {}


class SamplingParametersCallback(CustomLogger):
    """Inject sampling parameters before LiteLLM sends requests.

    Adds logprobs and top_logprobs to all requests.
    Only adds return_token_ids for vLLM-compatible backends (not OpenAI/Anthropic).
    """

    def __init__(self, add_return_token_ids: bool = False, add_logprobs: bool = False):
        super().__init__()
        self.add_return_token_ids = add_return_token_ids
        self.add_logprobs = add_logprobs

    async def async_pre_call_hook(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        data = kwargs.get("data") or (args[2] if len(args) > 2 else {})

        if self.add_logprobs:
            data["logprobs"] = True

        if self.add_return_token_ids:
            data["return_token_ids"] = True

        return data


class TracingCallback(CustomLogger):
    """Log LLM calls to tracer right after provider response.

    Uses LiteLLM's async_post_call_success_hook which fires at the proxy level,
    once per HTTP request, immediately before the response is sent to the user.
    This guarantees we log with the actual response object, while still being
    pre-send, and avoids duplicate logging from nested deployment calls.
    """

    def __init__(self, tracer: SqliteTracer, *, await_persistence: bool = False):
        super().__init__()
        self.tracer = tracer
        self._await_persistence = await_persistence

    async def _persist_trace(
        self,
        *,
        data: dict[str, Any],
        response: ModelResponse | ModelResponseStream | Any,
        latency_ms: float,
        model: str,
        messages: list[Any],
    ) -> Any:
        metadata = _extract_rllm_metadata(data if isinstance(data, dict) else {})
        if not metadata:
            logger.warning(
                "TracingCallback: missing RLLM metadata in LiteLLM payload; available keys=%s",
                sorted(data.keys()) if isinstance(data, dict) else type(data).__name__,
            )

        usage = getattr(response, "usage", None)
        tokens = {
            "prompt": getattr(usage, "prompt_tokens", 0) if usage else 0,
            "completion": getattr(usage, "completion_tokens", 0) if usage else 0,
            "total": getattr(usage, "total_tokens", 0) if usage else 0,
        }

        if hasattr(response, "model_dump"):
            response_payload: Any = response.model_dump()
        elif isinstance(response, dict):
            response_payload = response
        else:
            response_payload = {"text": str(response)}

        # Extract the response ID from the LLM provider to use as trace_id/context_id
        # This ensures the context_id matches the actual completion ID from the provider
        response_id = response_payload.get("id", None)

        # Extract session_uids and session_name from metadata (sent from client)
        session_uids = metadata.get("session_uids", None)
        session_name = metadata.get("session_name")

        log_kwargs = dict(
            name=f"proxy/{model}",
            model=model,
            input={"messages": messages},
            output=response_payload,
            metadata=metadata,
            session_name=session_name,
            latency_ms=latency_ms,
            tokens=tokens,
            trace_id=response_id,
            session_uids=session_uids,
        )

        if not session_uids:
            logger.warning(
                "TracingCallback: session_uids missing for model=%s response_id=%s metadata_keys=%s",
                model,
                response_id,
                sorted(metadata.keys()),
            )

        if self._await_persistence:
            await self.tracer.log_llm_call_sync(**log_kwargs)
        else:
            self.tracer.log_llm_call(**log_kwargs)

        # Return response unchanged
        return response

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: Any,
        response: ModelResponse | ModelResponseStream,
    ) -> Any:
        """Called once per HTTP request at proxy level, before response is sent to user."""

        model = data.get("model", "unknown") if isinstance(data, dict) else "unknown"
        messages = data.get("messages", []) if isinstance(data, dict) else []
        latency_ms = float(getattr(response, "response_ms", 0.0) or 0.0)
        return await self._persist_trace(
            data=data,
            response=response,
            latency_ms=latency_ms,
            model=model,
            messages=messages,
        )

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Fallback success hook for newer LiteLLM callback paths."""

        data = kwargs if isinstance(kwargs, dict) else {}
        model = data.get("model") or getattr(response_obj, "model", "unknown")
        messages = data.get("messages")
        if messages is None:
            standard_logging_object = data.get("standard_logging_object") or {}
            messages = standard_logging_object.get("messages", [])

        duration = end_time - start_time
        if isinstance(duration, timedelta):
            latency_ms = duration.total_seconds() * 1000.0
        else:
            latency_ms = float(duration) * 1000.0

        return await self._persist_trace(
            data=data,
            response=response_obj,
            latency_ms=latency_ms,
            model=model,
            messages=messages or [],
        )
