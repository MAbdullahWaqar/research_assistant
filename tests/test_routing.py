"""Unit tests for graph routing helpers (no API keys required)."""

from __future__ import annotations

from app.graph import (
    NODE_RESEARCH,
    NODE_SYNTHESIS,
    NODE_VALIDATOR,
    route_after_research,
    route_after_validator,
)
from app.config import CONFIDENCE_THRESHOLD, MAX_RESEARCH_ATTEMPTS


def test_route_after_research_high_confidence_skips_validator() -> None:
    assert route_after_research({"confidence_score": 9.0}) == NODE_SYNTHESIS
    assert route_after_research({"confidence_score": CONFIDENCE_THRESHOLD}) == NODE_SYNTHESIS


def test_route_after_research_low_confidence_goes_to_validator() -> None:
    assert route_after_research({"confidence_score": 5.0}) == NODE_VALIDATOR
    assert route_after_research({"confidence_score": CONFIDENCE_THRESHOLD - 0.1}) == NODE_VALIDATOR


def test_route_after_validator_insufficient_with_retries_left() -> None:
    state = {"validation_result": "insufficient", "research_attempts": 1}
    assert route_after_validator(state) == NODE_RESEARCH


def test_route_after_validator_sufficient() -> None:
    state = {"validation_result": "sufficient", "research_attempts": 1}
    assert route_after_validator(state) == NODE_SYNTHESIS


def test_route_after_validator_max_attempts_forces_synthesis() -> None:
    state = {
        "validation_result": "insufficient",
        "research_attempts": MAX_RESEARCH_ATTEMPTS,
    }
    assert route_after_validator(state) == NODE_SYNTHESIS
