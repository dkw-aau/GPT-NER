# encoding: utf-8
"""
Anthropic provider – backed by the ``anthropic`` Python SDK.

Install: pip install anthropic

Credentials
-----------
ANTHROPIC_API_KEY
    Required.
"""
import os
from typing import List, Tuple, Type

from base_provider import LLMProvider
from logger import get_logger

logger = get_logger(__name__)


class AnthropicProvider(LLMProvider):
    """
    Text-completion provider backed by the Anthropic Messages API.

    Anthropic's API is chat-based; each prompt is sent as a ``user``
    message and the model's reply is returned as the completion.

    Configuration keys (from config.yaml ``anthropic`` section)
    -----------------------------------------------------------
    max_tokens_to_sample : int
        Alias for ``max_tokens`` (kept for clarity).
    """

    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
    ):
        super().__init__(temperature=temperature, max_tokens=max_tokens, top_p=top_p)
        self.model = model

    @property
    def retryable_exceptions(self) -> Tuple[Type[Exception], ...]:
        try:
            import anthropic
            return (anthropic.RateLimitError, anthropic.APIStatusError)
        except ImportError:
            return (Exception,)

    def _complete(self, prompt_list: List[str]) -> List[str]:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required for the Anthropic provider. "
                "Install it with: pip install anthropic"
            ) from exc

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY environment variable is not set."
            )

        client = anthropic.Anthropic(api_key=api_key)

        results = []
        for prompt in prompt_list:
            message = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                messages=[{"role": "user", "content": prompt}],
            )
            results.append(message.content[0].text)

        logger.info(
            msg="prompt_and_result",
            extra={"prompt_list": prompt_list, "results": results},
        )
        return results
