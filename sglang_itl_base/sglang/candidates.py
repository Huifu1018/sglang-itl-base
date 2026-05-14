"""Candidate-row helpers for SGLang spec-v1 verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class CandidateRows:
    rows: tuple[tuple[int, ...], ...]
    draft_token_num: int
    proposed_target_tokens: int
    draft_prob_rows: tuple[object | None, ...] = ()


def build_linear_candidate_rows(
    roots: Sequence[int],
    target_rows: Sequence[Sequence[int]],
    *,
    max_draft_token_num: int,
    draft_prob_rows: Sequence[object | None] | None = None,
) -> CandidateRows:
    """Build equal-width linear verify rows.

    SGLang's target verifier expects a single row width for the whole batch.
    The width is therefore clipped to the shortest real candidate row in the
    batch; no fake pad/eos candidates are introduced.
    """

    if max_draft_token_num <= 0:
        raise ValueError("max_draft_token_num must be positive.")
    if len(roots) != len(target_rows):
        raise ValueError("roots and target_rows must have the same length.")
    if draft_prob_rows is not None and len(draft_prob_rows) != len(roots):
        raise ValueError("draft_prob_rows must have the same length as roots.")

    max_target_tokens = max(0, max_draft_token_num - 1)
    raw_rows: list[tuple[int, ...]] = []
    proposed_target_tokens = 0
    for root, targets in zip(roots, target_rows):
        clipped = tuple(int(token_id) for token_id in targets[:max_target_tokens])
        proposed_target_tokens += len(clipped)
        raw_rows.append((int(root), *clipped))

    if not raw_rows:
        return CandidateRows(rows=(), draft_token_num=1, proposed_target_tokens=0)

    draft_token_num = max(1, min(len(row) for row in raw_rows))
    draft_token_num = min(draft_token_num, max_draft_token_num)
    rows = tuple(row[:draft_token_num] for row in raw_rows)

    clipped_prob_rows: tuple[object | None, ...] = ()
    if draft_prob_rows is not None:
        clipped_prob_rows = tuple(draft_prob_rows)

    return CandidateRows(
        rows=rows,
        draft_token_num=draft_token_num,
        proposed_target_tokens=proposed_target_tokens,
        draft_prob_rows=clipped_prob_rows,
    )
