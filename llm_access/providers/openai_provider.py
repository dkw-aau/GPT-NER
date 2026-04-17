# encoding: utf-8
"""
OpenAI provider – supports both the legacy openai<1.0 API and the new
openai>=1.0 client API.
"""
import os
from typing import List, Tuple, Type

from base_provider import LLMProvider
from logger import get_logger

logger = get_logger(__name__)


def _detect_new_api() -> bool:
    """Return True if the installed openai package uses the v1+ client API."""
    try:
        import openai  # noqa: F401
        return hasattr(openai, "OpenAI")
    except ImportError:
        return False


class OpenAIProvider(LLMProvider):
    """
    Text-completion provider backed by the OpenAI API.

    Credentials
    -----------
    OPENAI_API_KEY
        Required.
    OPENAI_API_BASE  (optional)
        Override the API base URL (e.g. for Azure OpenAI or a proxy).

    Configuration keys (from config.yaml ``openai`` section)
    ---------------------------------------------------------
    best_of : int
        How many completions to generate server-side; best is returned.
    api_base : str
        Alternative base URL (also settable via env var OPENAI_API_BASE).
    api_type : str
        ``"azure"`` for Azure OpenAI.
    api_version : str
        API version string required by Azure.
    """

    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
        best_of: int = 1,
        api_base: str = None,
        api_type: str = None,
        api_version: str = None,
    ):
        super().__init__(temperature=temperature, max_tokens=max_tokens, top_p=top_p)
        self.model = model
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.best_of = best_of
        self.api_base = api_base or os.environ.get("OPENAI_API_BASE")
        self.api_type = api_type
        self.api_version = api_version
        self._use_new_api = _detect_new_api()

    # ------------------------------------------------------------------
    # Retryable exceptions differ between old and new SDK
    # ------------------------------------------------------------------

    @property
    def retryable_exceptions(self) -> Tuple[Type[Exception], ...]:
        try:
            if self._use_new_api:
                import openai
                return (openai.RateLimitError,)
            else:
                import openai
                return (openai.error.RateLimitError,)
        except Exception:
            return (Exception,)

    # ------------------------------------------------------------------
    # Core completion call
    # ------------------------------------------------------------------

    def _complete(self, prompt_list: List[str]) -> List[str]:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set."
            )

        if self._use_new_api:
            return self._complete_new(prompt_list, api_key)
        return self._complete_legacy(prompt_list, api_key)

    def _complete_new(self, prompt_list: List[str], api_key: str) -> List[str]:
        """openai >= 1.0 path."""
        import openai

        kwargs = dict(api_key=api_key)
        if self.api_base:
            kwargs["base_url"] = self.api_base

        if self.api_type == "azure":
            client = openai.AzureOpenAI(
                api_key=api_key,
                azure_endpoint=self.api_base,
                api_version=self.api_version,
            )
        else:
            client = openai.OpenAI(**kwargs)

        results = []
        for prompt in prompt_list:
            response = client.completions.create(
                model=self.model,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                n=self.best_of,
            )
            results.append(response.choices[0].text)

        logger.info(
            msg="prompt_and_result",
            extra={"prompt_list": prompt_list, "results": results},
        )
        return results

    def _complete_legacy(self, prompt_list: List[str], api_key: str) -> List[str]:
        """openai < 1.0 (legacy) path."""
        import openai

        openai.api_key = api_key
        if self.api_base:
            openai.api_base = self.api_base
        if self.api_type:
            openai.api_type = self.api_type
        if self.api_version:
            openai.api_version = self.api_version

        response = openai.Completion.create(
            engine=self.model,
            prompt=prompt_list,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            best_of=self.best_of,
        )
        results = [choice.text for choice in response.choices]
        logger.info(
            msg="prompt_and_result",
            extra={"prompt_list": prompt_list, "results": results},
        )
        return results
