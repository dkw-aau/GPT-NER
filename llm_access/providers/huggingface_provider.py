# encoding: utf-8
"""
HuggingFace provider – backed by ``huggingface_hub.InferenceClient``.

Install: pip install huggingface_hub

Credentials
-----------
HF_TOKEN
    Optional for public models; required for gated or private models.

Configuration keys (from config.yaml ``huggingface`` section)
-------------------------------------------------------------
endpoint_url : str
    Custom inference endpoint URL.  Defaults to the public
    Inference API (``https://api-inference.huggingface.co``).
task : str
    Inference task.  Use ``"conversational"`` for instruction/chat models
    (e.g. ``mistralai/Mistral-7B-Instruct-v0.2``, ``meta-llama/Meta-Llama-3-8B-Instruct``).
    Use ``"text-generation"`` for base (non-instruction) models.
    Defaults to ``"text-generation"``.
"""
import os
from typing import List, Tuple, Type

from base_provider import LLMProvider
from logger import get_logger

logger = get_logger(__name__)


class HuggingFaceProvider(LLMProvider):
    """
    Text-completion provider backed by the HuggingFace Inference API or a
    dedicated Inference Endpoint.
    """

    SUPPORTED_TASKS = ("text-generation", "conversational")

    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        endpoint_url: str = None,
        task: str = "text-generation",
    ):
        if task not in self.SUPPORTED_TASKS:
            raise ValueError(
                f"Unsupported HuggingFace task {task!r}. "
                f"Supported tasks: {', '.join(self.SUPPORTED_TASKS)}. "
                "Use 'conversational' for instruction/chat models and "
                "'text-generation' for base models."
            )
        super().__init__(temperature=temperature, max_tokens=max_tokens, top_p=top_p)
        self.model = model
        self.endpoint_url = endpoint_url
        self.task = task

    @property
    def retryable_exceptions(self) -> Tuple[Type[Exception], ...]:
        try:
            from huggingface_hub.utils import HfHubHTTPError
            return (HfHubHTTPError,)
        except ImportError:
            return (Exception,)

    def _complete(self, prompt_list: List[str]) -> List[str]:
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:
            raise ImportError(
                "The 'huggingface_hub' package is required for the HuggingFace "
                "provider.  Install it with: pip install huggingface_hub"
            ) from exc

        token = os.environ.get("HF_TOKEN")

        client = InferenceClient(
            model=self.endpoint_url or self.model,
            token=token,
        )

        results = []
        for prompt in prompt_list:
            if self.task == "conversational":
                # Many instruction-tuned models (e.g. Mistral-7B-Instruct-v0.2)
                # are only available under the "conversational" task on the HF
                # Inference API.  Use chat_completion() and wrap the prompt as a
                # user message.
                response = client.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=max(self.temperature, 1e-2),  # HF rejects 0.0
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                )
                results.append(response.choices[0].message.content)
            else:
                response = client.text_generation(
                    prompt=prompt,
                    temperature=max(self.temperature, 1e-2),  # HF rejects 0.0
                    max_new_tokens=self.max_tokens,
                    top_p=self.top_p,
                    do_sample=self.temperature > 0,
                )
                # InferenceClient.text_generation returns a str directly
                results.append(response if isinstance(response, str) else response.generated_text)

        logger.info(
            msg="prompt_and_result",
            extra={"prompt_list": prompt_list, "results": results},
        )
        return results
