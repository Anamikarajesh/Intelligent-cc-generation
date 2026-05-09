"""Reaction scoring helpers."""

from __future__ import annotations

from cc_suggester.core.types import ReactionResult


def strongest_signal(reaction: ReactionResult) -> str | None:
    """Return the strongest named reaction signal, if available."""

    if not reaction.reaction_signals:
        return None
    return max(reaction.reaction_signals, key=lambda key: reaction.reaction_signals[key])
