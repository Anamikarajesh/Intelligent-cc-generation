"""Combine audio and visual evidence into caption decisions."""

from __future__ import annotations

from cc_suggester.core.config import PipelineConfig
from cc_suggester.core.types import AudioEventCandidate, CaptionSuggestion, ReactionResult
from cc_suggester.decision.labels import caption_for
from cc_suggester.decision.rules import ambient_penalty, importance_prior, is_high_impact


def decide_captions(
    audio_events: list[AudioEventCandidate],
    reactions: list[ReactionResult],
    config: PipelineConfig,
) -> list[CaptionSuggestion]:
    """Create final caption suggestions from audio and visual signals."""

    reaction_by_key = {
        (reaction.event_id, reaction.start_time, reaction.end_time): reaction
        for reaction in reactions
    }
    suggestions: list[CaptionSuggestion] = []

    for event in audio_events:
        reaction = reaction_by_key.get((event.event_id, event.start_time, event.end_time))
        reaction_confidence = reaction.reaction_confidence if reaction else 0.0
        prior = importance_prior(event.event_id)
        penalty = ambient_penalty(event.event_id)
        score = _score(
            audio_confidence=event.audio_confidence,
            reaction_confidence=reaction_confidence,
            prior=prior,
            penalty=penalty,
        )

        accepted = score >= config.decision_threshold
        requires_review = config.review_threshold <= score < config.decision_threshold
        if is_high_impact(event.event_id) and event.audio_confidence >= 0.70:
            accepted = True
            requires_review = False

        reason = _reason_for(event, reaction_confidence, score, accepted, requires_review, penalty)
        suggestions.append(
            CaptionSuggestion(
                event_id=event.event_id,
                start_time=event.start_time,
                end_time=event.end_time,
                audio_confidence=event.audio_confidence,
                reaction_confidence=reaction_confidence,
                decision_score=round(score, 3),
                accepted=accepted,
                reason=reason,
                caption_text=caption_for(event.event_id, config.language),
                language=config.language,
                requires_review=requires_review,
                debug_info={
                    "importance_prior": prior,
                    "ambient_penalty": penalty,
                    "high_impact": is_high_impact(event.event_id),
                    "reaction_signals": reaction.reaction_signals if reaction else {},
                },
            )
        )
    return suggestions


def _score(
    audio_confidence: float,
    reaction_confidence: float,
    prior: float,
    penalty: float,
) -> float:
    raw = (0.52 * audio_confidence) + (0.34 * reaction_confidence) + prior - penalty
    return max(0.0, min(1.0, raw))


def _reason_for(
    event: AudioEventCandidate,
    reaction_confidence: float,
    score: float,
    accepted: bool,
    requires_review: bool,
    penalty: float,
) -> str:
    if accepted:
        if reaction_confidence >= 0.50:
            return (
                f"Accepted because {event.event_id} has strong audio confidence "
                "and visible scene reaction."
            )
        return f"Accepted because {event.event_id} is important and audio confidence is high."
    if requires_review:
        return (
            f"Needs review because {event.event_id} is plausible but the combined "
            f"decision score is borderline ({score:.2f})."
        )
    if penalty > 0:
        return f"Rejected because {event.event_id} appears ambient or low-impact."
    return f"Rejected because combined audio and reaction evidence is weak ({score:.2f})."
