"""Confidence/darkness score + separation metric (E3 extends with PR/ROC figures)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn.metrics import roc_auc_score

if TYPE_CHECKING:
    from septentrion.schemas import MatchedContact


def score_contacts(contacts: list[MatchedContact], max_dist_m: float) -> list[MatchedContact]:
    """Assign a match_score in [0,1]: 1 = confidently matched, 0 = confidently dark."""
    for c in contacts:
        if c.dist_m is None:
            c.match_score = 0.0
        else:
            c.match_score = max(0.0, 1.0 - c.dist_m / max_dist_m)
    return contacts


def separation_auc(contacts: list[MatchedContact], truth_matched: list[bool]) -> float:
    """ROC AUC of match_score against ground-truth matched/dark labels."""
    y_true = [1 if t else 0 for t in truth_matched]
    y_score = [c.match_score for c in contacts]
    if len(set(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))
