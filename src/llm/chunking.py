"""Intelligent chunking / truncation (Phase III).

Goal: NEVER trigger a 413 Payload Too Large, while retaining the most
semantically dense content.

Strategy:
    1. Count tokens with tiktoken (provider-agnostic approximation).
    2. If the document fits the model budget, return it unchanged.
    3. Otherwise, perform head + tail salience truncation. The beginning
       and end of an article/page often contain titles, metadata, dates,
       authors, conclusions, and other extraction-relevant information.
    4. Reserve token budget for the truncation marker itself.
    5. Enforce the token budget using the actual tokenizer as the final
       authority.

The returned string is guaranteed to contain no more than ``max_tokens``
tokens when ``max_tokens`` is positive.
"""

from __future__ import annotations

import tiktoken


_ENC = tiktoken.get_encoding("cl100k_base")
_TRUNCATION_MARKER = "\n\n[…truncated for length…]\n\n"


def count_tokens(text: str) -> int:
    """Return the number of cl100k_base tokens in ``text``."""
    return len(_ENC.encode(text))


def salient_truncate(
    text: str,
    *,
    max_tokens: int,
    head_ratio: float = 0.7,
) -> str:
    """Return ``text`` truncated to at most ``max_tokens`` tokens.

    When truncation is required, the function keeps content from both the
    beginning and end of the document while reserving enough budget for the
    truncation marker.

    Args:
        text: Input document.
        max_tokens: Maximum number of output tokens.
        head_ratio: Fraction of the available content budget assigned to
            the beginning of the document.

    Returns:
        The original text if it already fits, otherwise a head + marker +
        tail representation guaranteed to fit within ``max_tokens``.

    Raises:
        ValueError: If ``max_tokens`` is not positive or ``head_ratio`` is
            outside the inclusive range [0, 1].
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than 0")

    if not 0.0 <= head_ratio <= 1.0:
        raise ValueError("head_ratio must be between 0.0 and 1.0")

    tokens = _ENC.encode(text)

    if len(tokens) <= max_tokens:
        return text

    marker_tokens = _ENC.encode(_TRUNCATION_MARKER)

    # If the budget cannot even accommodate the marker, return the first
    # max_tokens directly. This is still guaranteed to respect the budget.
    if len(marker_tokens) >= max_tokens:
        return _ENC.decode(tokens[:max_tokens])

    content_budget = max_tokens - len(marker_tokens)

    head_n = int(content_budget * head_ratio)
    tail_n = content_budget - head_n

    head = _ENC.decode(tokens[:head_n]) if head_n else ""
    tail = _ENC.decode(tokens[-tail_n:]) if tail_n else ""

    result = f"{head}{_TRUNCATION_MARKER}{tail}"

    # Defensive final check. Tokenization of concatenated decoded strings
    # can differ slightly from independently encoded segments, so enforce
    # the contract against the actual final output.
    result_tokens = _ENC.encode(result)

    if len(result_tokens) > max_tokens:
        result = _ENC.decode(result_tokens[:max_tokens])

    return result