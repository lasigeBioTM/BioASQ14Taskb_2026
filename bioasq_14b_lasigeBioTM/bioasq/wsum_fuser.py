#wsum fuser

from ranx import Run, fuse
import logging

log=logging.getLogger(__name__)

FUSION_NORM= "min-max"
FUSION_METHOD= "wsum"
FUSION_WEIGHT_SPARSE= 0.83 #obtained via optimize_fusion() from ranx
FUSION_WEIGHT_DENSE= 0.17 


class WSumFuser:
    """
    Combines sparse (BM25) + dense (PubMedBERT) runs using weighted sum fusion via ranx:
    Note: ranx wsum params format : params={"weights": [w_sparse, w_dense]}.
    """

    def __init__(self, norm: str= FUSION_NORM, weight_sparse: float = FUSION_WEIGHT_SPARSE, weight_dense: float= FUSION_WEIGHT_DENSE):
        self.norm= norm
        self.weight_sparse= weight_sparse
        self.weight_dense= weight_dense
        
        log.info(f"WSumFuser norm={norm}, w_sparse={weight_sparse}, w_dense={weight_dense}")

    def combine(self, query_id: str, sparse_scores: dict[str, float], dense_scores: dict[str, float], top_n: int = 10) -> list[tuple[str, float]]:
        """
        Returns a sorted list of (pmid, fused_score) — best first, length ≤ top_n.
        """
        run_sparse = Run({query_id: sparse_scores}, name="sparse")
        run_dense  = Run({query_id: dense_scores},  name="dense")

        fused_run = fuse(runs= [run_sparse, run_dense], norm = self.norm, method = FUSION_METHOD, params = {"weights": [self.weight_sparse, self.weight_dense]})

        ranked = sorted(fused_run.run[query_id].items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_n] 