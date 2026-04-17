# encoding: utf-8
"""
Abstract base class for LLM providers and shared retry logic.
"""
import random
import time
from abc import ABC, abstractmethod
from math import ceil
from typing import List, Tuple, Type

from logger import get_logger

logger = get_logger(__name__)

INIT_DELAY = 1
EXPONENTIAL_BASE = 2
MAX_RETRIES = 6


class LLMProvider(ABC):
    """Provider-agnostic interface for text-completion LLMs."""

    delay: float = INIT_DELAY

    def __init__(self, temperature: float, max_tokens: int, top_p: float):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

    # ------------------------------------------------------------------
    # Subclasses implement this one method
    # ------------------------------------------------------------------

    @abstractmethod
    def _complete(self, prompt_list: List[str]) -> List[str]:
        """
        Send *prompt_list* to the LLM and return one completion string per
        prompt.  Raise provider-specific exceptions on transient errors.
        """

    # ------------------------------------------------------------------
    # Override in subclasses to declare which exceptions trigger a retry
    # ------------------------------------------------------------------

    @property
    def retryable_exceptions(self) -> Tuple[Type[Exception], ...]:
        """Exceptions that should trigger an exponential-backoff retry."""
        return (Exception,)

    # ------------------------------------------------------------------
    # Shared retry / back-off logic (not overridden by subclasses)
    # ------------------------------------------------------------------

    def complete(self, prompt_list: List[str], jitter: bool = True) -> List[str]:
        """
        Call ``_complete`` with automatic exponential-backoff retries.

        Parameters
        ----------
        prompt_list:
            Prompts to send to the LLM.
        jitter:
            When ``True``, add random jitter to the back-off delay.

        Returns
        -------
        List[str]
            One completion string per prompt.
        """
        num_retries = 0

        while True:
            used_delay = LLMProvider.delay
            try:
                logger.info(f"Delay={used_delay - 1}")
                for _ in range(ceil(max(used_delay - 1, 0))):
                    time.sleep(1)

                results = self._complete(prompt_list)
                LLMProvider.delay = INIT_DELAY
                return results

            except self.retryable_exceptions as exc:
                logger.info(f"Retryable error ({type(exc).__name__}): {exc}; retrying …")
                num_retries += 1
                if num_retries > MAX_RETRIES:
                    logger.error("Maximum retries exceeded.")
                    raise RuntimeError(
                        f"Maximum number of retries ({MAX_RETRIES}) exceeded."
                    ) from exc
                LLMProvider.delay = max(
                    LLMProvider.delay,
                    used_delay * EXPONENTIAL_BASE * (1 + jitter * random.random()),
                )

            except Exception as exc:
                logger.info(f"Unexpected error ({type(exc).__name__}): {exc}; retrying …")
                num_retries += 1
                if num_retries > MAX_RETRIES:
                    logger.error("Maximum retries exceeded.")
                    raise RuntimeError(
                        f"Maximum number of retries ({MAX_RETRIES}) exceeded."
                    ) from exc
                LLMProvider.delay = max(
                    LLMProvider.delay,
                    used_delay * EXPONENTIAL_BASE * (1 + jitter * random.random()),
                )

    # ------------------------------------------------------------------
    # Convenience alias used by legacy call sites
    # ------------------------------------------------------------------

    def get_multiple_sample(self, prompt_list: List[str], jitter: bool = True) -> List[str]:
        """Alias for :meth:`complete` for backward compatibility."""
        return self.complete(prompt_list, jitter=jitter)
