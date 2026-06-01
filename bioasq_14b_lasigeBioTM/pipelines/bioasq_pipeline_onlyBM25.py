"""
Pipeline:
  PISA BM25 → top-1/2000 → cross-encoder → X

Exemple of usage:
    python bioasq_pipeline_onlyBM25.py \
        --input   .../BioASQ-task14bPhaseA-testset4 \
        --output  .../submission_onlyBM25.json \
        --corpus  .../pubmed2026.lmdb \
        --pisa-index .../pubmed2026_pisa \
        --top-k-retrieval 1000 
"""
import argparse
import json
import logging
import sys
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

CROSS_ENCODER_MODEL = ""
RERANKER_DOCS_POOL= 1000

CE_THRESHOLD=0.91
CE_MIN_DOCS= 1

from bioasq.helpers import Snippet, PhaseAResult
from bioasq.corpus_store import CorpusStore
from bioasq.sparse_retriever import SparseRetriever
from bioasq.snippet_extractor import SnippetExtractor
from bioasq.thresholds import apply_threshold_cutoff

class CrossEncoderReranker:
    
    def __init__(self, model_name: str = CROSS_ENCODER_MODEL):
        log.info(f"A carregar cross-encoder: {model_name} ...")
        self.model = CrossEncoder(model_name, trust_remote_code=True)
        log.info("CrossEncoderReranker pronto.")

    def rerank_documents(self, query: str, candidates: list[tuple[str, float]], corpus: CorpusStore, top_n: int = 10) -> list[tuple[str, float]]:
        pairs, found_pmids, missing_pmids = [], [], []

        for pmid, _ in candidates:
            doc = corpus.get(pmid)
            if doc is None:
                missing_pmids.append(pmid)
                continue
            pairs.append((query, f"{doc.title} {doc.content}".strip()))
            found_pmids.append(pmid)

        if not pairs:
            return candidates[:top_n]

        scores = self.model.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(found_pmids, scores), key=lambda x: x[1], reverse=True)
        result = [(pmid, float(score)) for pmid, score in ranked]

        for pmid in missing_pmids:
            if len(result) >= top_n:
                break
            result.append((pmid, 0.0))

        return result

    def rerank_snippets(self, query: str, snippets: list[Snippet], top_n: int = 10) -> list[Snippet]:
        if not snippets:
            return []
        pairs = [(query, s.text) for s in snippets]
        scores = self.model.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(snippets, scores), key=lambda x: x[1], reverse=True)
        return [s for s, _ in ranked[:top_n]]

class BioASQSparseOnlyPipeline:
    """
    BM25-only pipeline: no FAISS, no fusion.
    BM25 top-1000 → cross-encoder top-100 → top-10 documents.
    """

    def __init__(
        self,
        corpus_path: str,
        pisa_index_path: str,
        top_k_retrieval: int = 1000,
        top_k_docs: int = 10,
        top_snippets: int = 10,
        reranker_pool: int = RERANKER_DOCS_POOL,
        ce_threshold: float = CE_THRESHOLD,
        ce_min_docs: int = CE_MIN_DOCS,
    ):

        self.corpus = CorpusStore(corpus_path)
        self.sparse = SparseRetriever(
            index_path = pisa_index_path,
            num_results = top_k_retrieval
        )
        bi_encoder = SentenceTransformer(EMBEDDING_MODEL)
        self.snippet_extractor = SnippetExtractor(bi_encoder)
        self.reranker = CrossEncoderReranker()

        self.top_k_retrieval = top_k_retrieval
        self.top_k_docs = top_k_docs
        self.top_snippets = top_snippets
        self.reranker_pool = reranker_pool
        self.ce_threshold= ce_threshold
        self.ce_min_docs= ce_min_docs

    def process_question(self, question: dict) -> PhaseAResult:
        qtype = question.get("type", "")
        qid = question["id"]
        qbody = question["body"]

        log.debug(f"Processing [{qid}]: {qbody[:80]}...")

        # Step 1: BM25 retrieval → top-1000
        sparse_scores = self.sparse.retrieve(qbody, question_type=qtype)

        if not sparse_scores:
            log.warning(f"[{qid}] BM25 returned 0 results — returning empty result.")
            return PhaseAResult(question_id=qid, question_type=qtype, documents=[], snippets=[])

        # Step 2: Sort by BM25 score → take top reranker_pool (100) for cross-encoder
        bm25_pool = sorted(sparse_scores.items(), key=lambda x: x[1], reverse=True)[:self.reranker_pool]

        # Step 3: Cross-encoder reranks top-100 → final top-10 documents
        reranked_all = self.reranker.rerank_documents(
            query = qbody,
            candidates = bm25_pool,
            corpus = self.corpus,
            top_n = len(bm25_pool)
        )

        rerank_docs= apply_threshold_cutoff(reranked=reranked_all, threshold=self.ce_threshold,
                                           min_docs= self.ce_min_docs, max_docs= self.top_k_docs)

        doc_urls = [PUBMED_URL_TEMPLATE.format(pmid=pmid) for pmid, _ in rerank_docs]

        # Step 4: Extract snippet candidates from the top-100 BM25 pool
        all_snippets: list[Snippet] = []
        for pmid, _ in bm25_pool:
            doc = self.corpus.get(pmid)
            if doc is None:
                log.warning(f"{pmid} não está no corpus")
                continue
            all_snippets.extend(
                self.snippet_extractor.extract(query=qbody, doc=doc, top_n=20)
            )

        # Step 5: Cross-encoder reranks snippets → final top-10
        sorted_snippets = self.reranker.rerank_snippets(
            query = qbody,
            snippets = all_snippets,
            top_n = self.top_snippets,
        )

        return PhaseAResult(
            question_id = qid,
            question_type = qtype,
            documents = doc_urls,
            snippets = sorted_snippets,
        )

    def run(self, input_path: str, output_path: str):
        with open(input_path, encoding="utf-8") as f:
            data = json.load(f)

        questions = data.get("questions", [])
        log.info(f"Nº de perguntas: {len(questions)} de {input_path} ...")

        output_questions = []
        for question in tqdm(questions, desc="Phase A (sparse)", unit="q"):
            result = self.process_question(question)
            output_questions.append(self._to_bioasq_dict(result))

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"questions": output_questions}, f, indent=2, ensure_ascii=False)

        log.info(f"--->>>> Submission saved: {output_path}")
        doc_counts = [len(q['documents']) for q in output_questions]
        log.info(f"--->>>> Questions: {len(output_questions)}")
        log.info(f"--->>>> Avg docs: {sum(doc_counts)/len(doc_counts):.1f}  min={min(doc_counts)}  max={max(doc_counts)}")
        log.info(f"--->>>> Snippets: {sum(len(q['snippets']) for q in output_questions)}")

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
                    "offsetInEndSection": s.offset_in_end,
                }
                for s in result.snippets
            ],
        }


def parse_args():
    p = argparse.ArgumentParser(
        description="BioASQ Task 14b Phase A — BM25-only pipeline."
    )
    p.add_argument("--input", required=True, help="BioASQ batch JSON (questions only)")
    p.add_argument("--output", required=True, help="Output submission JSON")
    p.add_argument("--ce-threshold", type=float, default=CE_THRESHOLD, help="Set to -999 to disable (pure top-N)")
    p.add_argument("--ce-min-docs", type=int, default=CE_MIN_DOCS)
    p.add_argument("--corpus", required=True, help="PubMed JSONL corpus dir or file")
    p.add_argument("--pisa-index", required=True, help="PISA index directory")
    p.add_argument("--top-k-retrieval", type=int, default=1000, help="BM25 top-K (default: 1000)")
    p.add_argument("--top-k-docs", type=int, default=10, help="Docs returned by cross-encoder per system")
    p.add_argument("--top-snippets", type=int, default=10, help="Final top-N snippets (default: 10)")
    p.add_argument("--reranker-pool", type=int, default=RERANKER_DOCS_POOL, help=f"Candidates fed to cross-encoder (default: {RERANKER_DOCS_POOL})")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.top_k_docs > 10:
        log.info(
            f"--top-k-docs={args.top_k_docs}: CE will return top-{args.top_k_docs} per system. "
            "Pass --top-docs 10 to ensemble_outputs.py to cut to final top-10.")

    pipeline = BioASQSparseOnlyPipeline(
        corpus_path = args.corpus,
        pisa_index_path = args.pisa_index,
        top_k_retrieval = args.top_k_retrieval,
        top_k_docs = args.top_k_docs,
        top_snippets = args.top_snippets,
        reranker_pool = args.reranker_pool,
        ce_threshold= args.ce_threshold if args.ce_threshold > -999 else None,
        ce_min_docs = args.ce_min_docs
    )

    pipeline.run(input_path=args.input, output_path=args.output)

if __name__ == "__main__":
    main()