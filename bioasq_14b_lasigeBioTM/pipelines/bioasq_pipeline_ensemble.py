"""
Two output files:
  1. submission_ensemble.json  — BioASQ-compliant Phase A JSON
  2. scores_per_model.json — raw CE scores per model per question, for analysis and comparisons

Pipeline:
  BM25 (query expanded) → top-1000
           ↓
  CE model 1 scores top-300 candidates → per-doc scores + top-N docs/snippets
  CE model 2 scores top-300 candidates → per-doc scores + top-N docs/snippets
  CE model 3 scores top-300 candidates → per-doc scores + top-N docs/snippets
           ↓
  ranx RRF over per-model top-N lists → final top-n

Exemple of usage:
    python Ensemble_crossencoders.py \
        --input .../BioASQ-task14bPhaseA-testset4 \
        --output submission_ensemble.json \
        --scores-output scores_per_model.json \
        --corpus .../pubmed2026.lmdb \
        --pisa-index .../pubmed2026_pisa \
        --ce-models "model1" "model2" "model3" \
        --ce-top-n 100 \
        --reranker-pool 300 \
        --rrf-k 60
"""

import argparse
import json
import logging
import sys

from ranx import Run, fuse
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

PUBMED_URL_TEMPLATE = "http://www.ncbi.nlm.nih.gov/pubmed/{pmid}"
EMBEDDING_MODEL = "NeuML/pubmedbert-base-embeddings-matryoshka"

DEFAULT_RERANKER_POOL = 300
DEFAULT_CE_TOP_N = 50
DEFAULT_RRF_K = 10
DEFAULT_MIN_DOCS = 1     
DEFAULT_1_THRESHOLD = 0.91
DEFAULT_2_THRESHOLD = 0.65   
DEFAULT_3_THRESHOLD  = 4.0 
DEFAULT_1_LABEL = "ft"
DEFAULT_2_LABEL = "bge"
DEFAULT_3_LABEL = "minilm"
QID = "q"   # dummy ranx query id for single-question fusion

from bioasq.helpers import Snippet, PhaseAResult, ModelScores
from bioasq.corpus_store import CorpusStore
from bioasq.sparse_retriever import SparseRetriever
from bioasq.snippet_extractor import SnippetExtractor


class MultiCEReranker:
    """
    Loads the cross-encoder models and scores the same candidate pool with all of them in one pass per question.
    """

    def __init__(self, model_names: list[str], model_labels: list[str]):
        self.models: list[tuple[str, CrossEncoder]] = []
        for name, label in zip(model_names, model_labels):
            log.info(f"Loading CE [{label}]: {name} ...")
            ce = CrossEncoder(name, trust_remote_code=True)
            self.models.append((label, ce))
        log.info(f"MultiCEReranker ready — {len(self.models)} models.")

    def score_documents(
        self,
        query: str,
        candidates: list[tuple[str, float]],
        corpus: CorpusStore,
        top_n: int
    ) -> list[ModelScores]:
        
        pairs: list[tuple[str, str]] = []
        found_pmids: list[str] = []
        missing_pmids: list[str] = []

        for pmid, _ in candidates:
            doc = corpus.get(pmid)
            if doc is None:
                missing_pmids.append(pmid)
                continue
            pairs.append((query, f"{doc.title} {doc.content}".strip()))
            found_pmids.append(pmid)

        results: list[ModelScores] = []

        for label, ce in self.models:
            ms = ModelScores(model_name=label)

            if not pairs:
                results.append(ms)
                continue

            raw_scores = ce.predict(
                [(q, d) for q, d in pairs], show_progress_bar=False)

            for pmid, score in zip(found_pmids, raw_scores):
                url = PUBMED_URL_TEMPLATE.format(pmid=pmid)
                ms.doc_scores[url] = float(score)

            ranked = sorted(ms.doc_scores.items(), key=lambda x: x[1], reverse=True)
            ms.top_docs = [url for url, _ in ranked[:top_n]]

            for pmid in missing_pmids:
                url = PUBMED_URL_TEMPLATE.format(pmid=pmid)
                if url not in ms.doc_scores:
                    ms.doc_scores[url] = 0.0
                if len(ms.top_docs) < top_n:
                    ms.top_docs.append(url)

            results.append(ms)

        return results

    def score_snippets(
        self,
        query: str,
        snippets: list[Snippet],
        top_n: int
    ) -> list[ModelScores]:

        results: list[ModelScores] = []

        if not snippets:
            return [ModelScores(model_name=label) for label, _ in self.models]

        pairs = [(query, s.text) for s in snippets]

        for label, ce in self.models:
            ms = ModelScores(model_name=label)

            raw_scores = ce.predict(pairs, show_progress_bar=False)

            for snip, score in zip(snippets, raw_scores):
                ms.snip_scores[snip.key()] = float(score)

            ranked = sorted(
                zip(snippets, raw_scores), key=lambda x: x[1], reverse=True
            )
            ms.top_snippets = [s for s, _ in ranked[:top_n]]

            results.append(ms)

        return results


def apply_threshold_cutoff(
    ranked_items: list,
    scores_by_model:  dict[str, dict[str, float]],
    one_threshold: float = DEFAULT_1_THRESHOLD,
    two_threshold: float = DEFAULT_2_THRESHOLD,
    three_threshold: float = DEFAULT_3_THRESHOLD,
    one_label: str = DEFAULT_1_LABEL,
    two_label: str = DEFAULT_2_LABEL,
    three_label: str = DEFAULT_3_LABEL,
    min_docs: int = 1,
    max_docs: int = 10
) -> list:
    
    one_scores = scores_by_model.get(one_label, {})
    two_scores = scores_by_model.get(two_label, {})
    three_scores = scores_by_model.get(three_label, {})

    selected = []
    for item in ranked_items:
        url = item[0]
        passes = True
        if one_threshold is not None and one_scores.get(url, 0.0) < one_threshold:
            passes = False
        if two_threshold is not None and two_scores.get(url, 0.0) < two_threshold:
            passes = False
        if three_threshold is not None and three_scores.get(url, 0.0) < three_threshold:
            passes = False
        if passes:
            selected.append(item)

    if len(selected) < min_docs:
        log.debug(
            f"Combined threshold returned {len(selected)} docs — "
            f"falling back to top-{min_docs} from RRF ranking."
        )
        selected = list(ranked_items[:min_docs])

    return selected[:max_docs]


def rrf_fuse_docs(
    model_scores: list[ModelScores],
    rrf_k: int,
    top_n: int,
    one_threshold: float = DEFAULT_1_THRESHOLD,
    two_threshold: float = DEFAULT_2_THRESHOLD,
    three_threshold: float = DEFAULT_3_THRESHOLD,
    min_docs: int = DEFAULT_MIN_DOCS
) -> list[str]:
    runs = []
    for ms in model_scores:
        if not ms.top_docs:
            continue
        run_dict = {
            QID: {url: 1.0 / rank for rank, url in enumerate(ms.top_docs, start=1)}
        }
        runs.append(Run(run_dict, name=ms.model_name))

    if not runs:
        return []
    if len(runs) == 1:
        ranked = sorted(runs[0].run[QID].items(), key=lambda x: x[1], reverse=True)
    else:
        fused  = fuse(runs=runs, method="rrf", params={"k": rrf_k})
        ranked = sorted(fused.run[QID].items(), key=lambda x: x[1], reverse=True)

    scores_by_model = {ms.model_name: ms.doc_scores for ms in model_scores}

    selected = apply_threshold_cutoff(
        ranked_items = ranked,
        scores_by_model  = scores_by_model,
        one_threshold = one_threshold,
        two_threshold = two_threshold,
        three_threshold =three_threshold,
        min_docs = min_docs,
        max_docs = top_n
    )
    return [url for url, _ in selected]


def rrf_fuse_snippets(model_scores: list[ModelScores], rrf_k:int, top_n: int) -> list[Snippet]:
    snippet_lookup: dict[str, Snippet] = {}
    for ms in model_scores:
        for snip in ms.top_snippets:
            if snip.key() not in snippet_lookup:
                snippet_lookup[snip.key()] = snip

    runs = []
    for ms in model_scores:
        if not ms.top_snippets:
            continue
        run_dict = {
            QID: {
                snip.key(): 1.0 / rank
                for rank, snip in enumerate(ms.top_snippets, start=1)
            }
        }
        runs.append(Run(run_dict, name=ms.model_name))

    if not runs:
        return []
    if len(runs) == 1:
        ranked = sorted(runs[0].run[QID].items(), key=lambda x: x[1], reverse=True)
        return [snippet_lookup[k] for k, _ in ranked[:top_n] if k in snippet_lookup]

    fused = fuse(runs=runs, method="rrf", params={"k": rrf_k})
    ranked = sorted(fused.run[QID].items(), key=lambda x: x[1], reverse=True)
    return [snippet_lookup[k] for k, _ in ranked[:top_n] if k in snippet_lookup]

class BioASQMultiCEEnsemble:

    def __init__(
        self,
        corpus_path: str,
        pisa_index_path: str,
        ce_model_names: list[str],
        ce_model_labels: list[str],
        top_k_retrieval: int = 1000,
        top_k_docs: int = 10,
        top_snippets: int = 10,
        reranker_pool: int = DEFAULT_RERANKER_POOL,
        ce_top_n: int = DEFAULT_CE_TOP_N,
        rrf_k: int = DEFAULT_RRF_K,
        one_threshold: float = DEFAULT_1_THRESHOLD,
        two_threshold: float = DEFAULT_2_THRESHOLD,
        three_threshold: float = DEFAULT_3_THRESHOLD,
        min_docs: int = DEFAULT_MIN_DOCS
    ):

        self.corpus = CorpusStore(corpus_path)
        self.sparse = SparseRetriever(index_path = pisa_index_path, num_results = top_k_retrieval)

        bi_encoder = SentenceTransformer(EMBEDDING_MODEL)
        self.snippet_extractor = SnippetExtractor(bi_encoder)

        self.reranker = MultiCEReranker(ce_model_names, ce_model_labels)

        self.top_k_docs = top_k_docs
        self.top_snippets = top_snippets
        self.reranker_pool = reranker_pool
        self.ce_top_n = ce_top_n
        self.rrf_k = rrf_k
        self.one_threshold = one_threshold
        self.two_threshold = two_threshold
        self.three_threshold = three_threshold
        self.min_docs = min_docs
        self.ce_labels = ce_model_labels

    def process_question(self, question: dict) -> tuple[PhaseAResult, dict]:
        """
        Returns (PhaseAResult, scores_record).
        scores_record contains raw CE scores per model for analysis.
        """
        qtype = question.get("type", "")
        qid = question["id"]
        qbody = question["body"]

        bm25_scores = self.sparse.retrieve(qbody, question_type=qtype)

        if not bm25_scores:
            log.warning(f"[{qid}] BM25 returned 0 results.")
            empty = PhaseAResult(question_id=qid, question_type=qtype,
                                 documents=[], snippets=[])
            return empty, {"id": qid, "type": qtype, "models": {}}

        bm25_pool = sorted(
            bm25_scores.items(), key=lambda x: x[1], reverse=True
        )[:self.reranker_pool]

        doc_model_scores = self.reranker.score_documents(
            query = qbody,
            candidates = bm25_pool,
            corpus = self.corpus,
            top_n = self.ce_top_n)

        all_snippets: list[Snippet] = []
        for pmid, _ in bm25_pool:
            doc = self.corpus.get(pmid)
            if doc is None:
                continue
            all_snippets.extend(
                self.snippet_extractor.extract(query=qbody, doc=doc, top_n=20)
            )

        snip_model_scores = self.reranker.score_snippets(
            query = qbody,
            snippets = all_snippets,
            top_n = self.ce_top_n
        )

        final_docs = rrf_fuse_docs(
            doc_model_scores,
            rrf_k = self.rrf_k,
            top_n = self.top_k_docs,
            one_threshold = self.one_threshold,
            two_threshold = self.two_threshold,
            three_threshold = self.three_threshold,
            min_docs = self.min_docs
        )
        final_snips = rrf_fuse_snippets(snip_model_scores, rrf_k=self.rrf_k, top_n=self.top_snippets)

        result = PhaseAResult(
            question_id=qid, question_type=qtype,
            documents=final_docs, snippets=final_snips
        )

        scores_record = {
            "id": qid,
            "type": qtype,
            "body": qbody,
            "bm25_pool_size": len(bm25_pool),
            "models": {}
        }
        for ms_doc, ms_snip in zip(doc_model_scores, snip_model_scores):
            scores_record["models"][ms_doc.model_name] = {
                "doc_scores": ms_doc.doc_scores,
                "snip_scores": ms_snip.snip_scores,
                "top_docs": ms_doc.top_docs
            }

        return result, scores_record

    def run(self, input_path: str, output_path: str, scores_path: str):
        with open(input_path, encoding="utf-8") as f:
            data = json.load(f)
        questions = data.get("questions", [])
        log.info(f"len questions: {len(questions)}")

        output_questions = []
        all_scores = []

        for question in tqdm(questions, desc="Multi-CE Ensemble", unit="q"):
            result, scores_record = self.process_question(question)
            output_questions.append(self._to_bioasq_dict(result))
            all_scores.append(scores_record)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"questions": output_questions}, f, indent=2, ensure_ascii=False)
        log.info(f"Submission saved → {output_path}")
        log.info(f"Questions : {len(output_questions)}")
        doc_counts = [len(q['documents']) for q in output_questions]
        log.info(f"Avg docs  : {sum(doc_counts)/len(doc_counts):.1f}  "
                 f"min={min(doc_counts)}  max={max(doc_counts)}")
        log.info(f"Avg snips : {sum(len(q['snippets']) for q in output_questions)/len(output_questions):.1f}")

        with open(scores_path, "w", encoding="utf-8") as f:
            json.dump(all_scores, f, indent=2, ensure_ascii=False)
        log.info(f"Scores saved  → {scores_path}")

    @staticmethod
    def _to_bioasq_dict(result: PhaseAResult) -> dict:
        return {
            "id": result.question_id,
            "type": result.question_type,
            "documents": result.documents,
            "snippets": [
                {
                    "document": s.document,
                    "text": s.text,
                    "beginSection": s.begin_section,
                    "endSection": s.end_section,
                    "offsetInBeginSection": s.offset_in_begin,
                    "offsetInEndSection": s.offset_in_end
                }
                for s in result.snippets
            ],
        }

def parse_args():
    p = argparse.ArgumentParser(
        description="BioASQ Phase A — Multi-CE ensemble with score saving."
    )
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True,  help="BioASQ submission JSON")
    p.add_argument("--scores-output",  required=True,  help="Per-model scores JSON")
    p.add_argument("--corpus", required=True,  help="LMDB corpus path")
    p.add_argument("--pisa-index", required=True,  help="PISA index directory")
    p.add_argument("--ce-models", required=True, nargs="+", help="CE model names or paths (space-separated)")
    p.add_argument("--ce-names", required=True, nargs="+", help="Short labels for each CE model (must match --ce-models count)")
    p.add_argument("--top-k-retrieval",type=int, default=2000)
    p.add_argument("--top-k-docs", type=int, default=10, help="Final top-N docs in submission (default: 10)")
    p.add_argument("--top-snippets", type=int, default=10, help="Final top-N snippets in submission (default: 10)")
    p.add_argument("--reranker-pool", type=int, default=DEFAULT_RERANKER_POOL, help=f"BM25 candidates fed to CEs (default: {DEFAULT_RERANKER_POOL})")
    p.add_argument("--ce-top-n", type=int, default=DEFAULT_CE_TOP_N, help=f"Docs/snippets each CE keeps before RRF (default: {DEFAULT_CE_TOP_N})")
    p.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K, help=f"RRF constant k (default: {DEFAULT_RRF_K})")
    p.add_argument("--one-threshold", type=float, default=DEFAULT_1_THRESHOLD, help=f"score threshold (default: {DEFAULT_1_THRESHOLD}). Set to -999 to disable.")
    p.add_argument("--two-threshold", type=float, default=DEFAULT_2_THRESHOLD, help=f"score threshold (default: {DEFAULT_2_THRESHOLD}). Set to -999 to disable.")
    p.add_argument("--three-threshold", type=float, default=DEFAULT_3_THRESHOLD, help=f"score threshold (default: {DEFAULT_3_THRESHOLD}). Set to -999 to disable.")
    p.add_argument("--min-docs", type=int, default=DEFAULT_MIN_DOCS, help=f"Min docs always returned per question (default: {DEFAULT_MIN_DOCS})")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if len(args.ce_models) != len(args.ce_names):
        print("ERROR: --ce-models and --ce-names must have the same number of entries.")
        sys.exit(1)

    pipeline = BioASQMultiCEEnsemble(
        corpus_path = args.corpus,
        pisa_index_path = args.pisa_index,
        ce_model_names = args.ce_models,
        ce_model_labels = args.ce_names,
        top_k_retrieval = args.top_k_retrieval,
        top_k_docs = args.top_k_docs,
        top_snippets = args.top_snippets,
        reranker_pool = args.reranker_pool,
        ce_top_n  = args.ce_top_n,
        rrf_k = args.rrf_k,
        one_threshold = args.one_threshold if args.one_threshold > -999 else None,
        two_threshold = args.two_threshold if args.two_threshold > -999 else None,
        three_threshold = args.three_threshold if args.three_threshold > -999 else None,
        min_docs = args.min_docs
    )

    pipeline.run(
        input_path = args.input,
        output_path = args.output,
        scores_path = args.scores_output
    )

if __name__ == "__main__":
    main()