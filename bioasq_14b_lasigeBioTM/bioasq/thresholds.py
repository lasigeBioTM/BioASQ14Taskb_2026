#thresholds

"""
Filters CE-reranked (pmid, score) list keeping only docs where score >= threshold.
Fallback: if threshold removes everything, returns the top min_docs docs from the CE ranking so the submission is never empty.
"""

import logging

log=logging.getLogger(__name__)

def apply_threshold_cutoff(
    reranked:list[tuple[str, float]],
    threshold:float,
    min_docs:int =1,
    max_docs:int = 10
) -> list[tuple[str, float]]:
    
    selected = [(pmid, score) for pmid, score in reranked if score >= threshold]

    if len(selected) < min_docs:
        log.debug(
            f"Treshold {threshold} returned {len(selected)} docs — "
            f"falling back to top-{min_docs} from CE ranking."
        )
        selected = list(reranked[:min_docs])

    return selected[:max_docs]
