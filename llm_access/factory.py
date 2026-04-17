# encoding: utf-8
"""
Factory that maps the ``provider`` config key to a concrete
:class:`~base_provider.LLMProvider` instance.

Usage
-----
::

    import yaml
    from factory import get_provider

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    provider = get_provider(config)
    results = provider.complete(["Hello, world!"])
"""
import sys
import os

from base_provider import LLMProvider

# Allow sibling ``providers/`` sub-package to import from the same dir
_here = os.path.dirname(os.path.abspath(__file__))
_providers_dir = os.path.join(_here, "providers")
if _providers_dir not in sys.path:
    sys.path.insert(0, _providers_dir)


_PROVIDER_MAP = {
    "openai": "openai_provider.OpenAIProvider",
    "anthropic": "anthropic_provider.AnthropicProvider",
    "huggingface": "huggingface_provider.HuggingFaceProvider",
}


def get_provider(config: dict) -> LLMProvider:
    """
    Instantiate the LLM provider described by *config*.

    Parameters
    ----------
    config:
        Parsed contents of ``config.yaml``.  Must contain at least a
        ``provider`` key.

    Returns
    -------
    LLMProvider
        Ready-to-use provider instance.

    Raises
    ------
    ValueError
        If the ``provider`` value is not recognised.
    """
    provider_name = config.get("provider", "openai").lower()
    if provider_name not in _PROVIDER_MAP:
        supported = ", ".join(_PROVIDER_MAP)
        raise ValueError(
            f"Unknown provider '{provider_name}'. Supported providers: {supported}"
        )

    module_name, class_name = _PROVIDER_MAP[provider_name].rsplit(".", 1)

    import importlib
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)

    # Common parameters
    common = dict(
        model=config.get("model", ""),
        temperature=float(config.get("temperature", 0.0)),
        max_tokens=int(config.get("max_tokens", 512)),
        top_p=float(config.get("top_p", 1.0)),
    )

    # Provider-specific extra parameters
    provider_cfg = config.get(provider_name, {}) or {}

    if provider_name == "openai":
        kwargs = {
            **common,
            "frequency_penalty": float(config.get("frequency_penalty", 0)),
            "presence_penalty": float(config.get("presence_penalty", 0)),
            "best_of": int(provider_cfg.get("best_of", 1)),
            "api_base": provider_cfg.get("api_base"),
            "api_type": provider_cfg.get("api_type"),
            "api_version": provider_cfg.get("api_version"),
        }
    elif provider_name == "huggingface":
        kwargs = {
            **common,
            "endpoint_url": provider_cfg.get("endpoint_url"),
            "task": provider_cfg.get("task", "text-generation"),
        }
    else:
        kwargs = common

    return cls(**kwargs)
