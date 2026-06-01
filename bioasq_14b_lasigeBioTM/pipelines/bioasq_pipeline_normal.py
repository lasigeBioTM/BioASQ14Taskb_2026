"""
FAISS dense  → top-1000 ──┐
                          ├── WSum fusion → top-n → cross-encoder → final list
PISA BM25    → top-1000 ──┘  

Example of usage:
    python bioasq_pipeline_ctreshold.py \
        --input   .../BioASQ-task14bPhaseA-testset4 \
        --output .../submission.json \
        --faiss-index     .../pubmed2026.index \
        --corpus          .../pubmed2026.lmdb \
        --pisa-index      .../pubmed2026_pisa \
        --top-k-retrieval 1000 \
        --top-k-docs      10 \
        --top-snippets    10 \
        --reranker-pool 500 \
        --ce-threshold 0.91
"""

import argparse
import json
import logging
import sys
import faiss
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


PUBMED_URL_TEMPLATE= "http://www.ncbi.nlm.nih.gov/pubmed/{pmid}"
EMBEDDING_MODEL= "NeuML/pubmedbert-base-embeddings-matryoshka" 

FAISS_NPROBE= 512

CROSS_ENCODER_MODEL= "" 
RERANKER_DOCS_POOL= 500  

CE_THRESHOLD = 0.91
CE_MIN_DOCS  = 1
 
from bioasq.helpers import Snippet, PhaseAResult
from bioasq.corpus_store import CorpusStore      
from bioasq.snippet_extractor import SnippetExtractor
from bioasq.thresholds import apply_threshold_cutoff
from bioasq.wsum_fuser import WSumFuser
from bioasq.sparse_retriever import SparseRetriever
 
class DenseRetriever:

    def __init__(self, index_path: str, model_name: str = EMBEDDING_MODEL):
        
        log.info(f"Loading FAISS index from {index_path} ...")
        
        self.index = faiss.read_index(index_path)
        self.index.nprobe = FAISS_NPROBE
        
        log.info(
            f"FAISS index: {self.index.ntotal:,} vetores, "
            f"nprobe={self.index.nprobe}"
        )

        log.info(f"carregar embedding model: {model_name} ...")
        self.model = SentenceTransformer(model_name)
        log.info("DenseRetriever readyyyyy")

    def retrieve(self, query: str, top_k: int = 100) -> dict[str, float]:
        """
        Returns {pmid_str: inner_product_score} — higher is better.
        PMIDs come back as int64 directly from the FAISS index.
        """
        vec = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, pmid_ints = self.index.search(vec, top_k)
        results: dict[str, float] = {}
        for score, pmid_int in zip(scores[0], pmid_ints[0]):
            if pmid_int < 0:        # FAISS returns -1 for empty slots
                continue
            results[str(pmid_int)] = float(score)
        return results

class CrossEncoderReranker:
    
    def __init__(self, model_name: str= CROSS_ENCODER_MODEL):
        log.info(f"carregar o cross-encoder: {model_name} ...")
        self.model = CrossEncoder(model_name, trust_remote_code=True)
        log.info("CrossEncoderReranker readyyyy")

    def rerank_documents(self, query: str, candidates: list[tuple[str, float]], corpus: "CorpusStore", top_n: int = 10) -> list[tuple[str, float]]:
    
        pairs: list[tuple[str, str]] = []
        found_pmids: list[str]= []
        missing_pmids: list[str]= []

        for pmid, _ in candidates:
            doc = corpus.get(pmid)
            if doc is None:
                missing_pmids.append(pmid)
                continue
            doc_text = f"{doc.title} {doc.content}".strip()
            pairs.append((query, doc_text))
            found_pmids.append(pmid)

        if not pairs:
            return candidates[:top_n]

        scores = self.model.predict(pairs, show_progress_bar=False)

        ranked = sorted(zip(found_pmids, scores), key=lambda x: x[1], reverse=True)

        result = [(pmid, float(score)) for pmid, score in ranked]
        for pmid in missing_pmids:
            result.append((pmid, 0.0))

        return result

    def rerank_snippets(self, query:str, snippets: list["Snippet"], top_n: int = 10) -> list["Snippet"]:
    
        if not snippets:
            return []

        pairs=[(query, s.text) for s in snippets]
        scores=self.model.predict(pairs, show_progress_bar=False)

        ranked= sorted( zip(snippets, scores), key=lambda x: x[1], reverse=True)
        return [snippet for snippet, _ in ranked[:top_n]]

class BioASQPhaseAPipeline:
    def __init__(
        self,
        faiss_index_path: str,
        corpus_path: str,
        pisa_index_path: str,
        top_k_retrieval: int = 1000,
        top_k_docs: int = 10,
        top_snippets: int = 10,
        reranker_pool: int = RERANKER_DOCS_POOL,
        ce_threshold: float = CE_THRESHOLD,
        ce_min_docs: int = CE_MIN_DOCS
    ):
        self.corpus = CorpusStore(corpus_path)

        self.dense  = DenseRetriever(faiss_index_path)
        self.sparse = SparseRetriever(
            index_path= pisa_index_path,
            num_results= top_k_retrieval
        )

        self.snippet_extractor = SnippetExtractor(self.dense.model)

        self.fuser = WSumFuser()

        self.reranker = CrossEncoderReranker()

        self.top_k_retrieval = top_k_retrieval
        self.top_k_docs = top_k_docs
        self.top_snippets = top_snippets
        self.reranker_pool = reranker_pool
        self.ce_threshold = ce_threshold
        self.ce_min_docs = ce_min_docs

    def process_question(self, question: dict) -> PhaseAResult:
        qtype=question.get("type", "")
        qid = question["id"]
        qbody = question["body"]

        log.debug(f"Processing [{qid}]: {qbody[:80]}...")

        dense_scores= self.dense.retrieve(qbody, top_k=self.top_k_retrieval)

        sparse_scores= self.sparse.retrieve(qbody, question_type=qtype)

        fused_pool= self.fuser.combine(query_id=qid, sparse_scores= sparse_scores, dense_scores= dense_scores, top_n= self.reranker_pool)

        if not fused_pool:
            log.warning(f"[{qid}] Fusion returned 0 candidates — returning empty result.")
            return PhaseAResult(question_id=qid, question_type=qtype, documents=[], snippets=[])

        reranked_all = self.reranker.rerank_documents(
            query = qbody,
            candidates = fused_pool,
            corpus = self.corpus,
            top_n = len(fused_pool)
        )

        reranked_docs = apply_threshold_cutoff(
            reranked = reranked_all,
            threshold = self.ce_threshold,
            min_docs = self.ce_min_docs,
            max_docs = self.top_k_docs
        )

        doc_urls: list[str] = [
            PUBMED_URL_TEMPLATE.format(pmid=pmid)
            for pmid, _ in reranked_docs]

        all_snippets: list[Snippet] = []
        
        for pmid, _ in fused_pool:
        
        #all_retrieval_pmids=list(dict.fromkeys(list(dense_scores.keys())+list(sparse_scores.keys())))
        #for pmid in all_retrieval_pmids:
            doc = self.corpus.get(pmid)
            if doc is None:
                log.warning(f"{pmid} não está no corpus")
                continue
            snippets = self.snippet_extractor.extract(query= qbody, doc= doc,top_n= 20)
            all_snippets.extend(snippets)

        sorted_snippets = self.reranker.rerank_snippets(query= qbody, snippets= all_snippets, top_n= self.top_snippets)

        return PhaseAResult(question_id= qid, question_type=qtype ,documents= doc_urls, snippets= sorted_snippets)

    def run(self, input_path: str, output_path: str):
        with open(input_path, encoding="utf-8") as f:
            data = json.load(f)

        questions = data.get("questions", [])
        log.info(f"len questions: {len(questions)} de {input_path} ...")

        output_questions = []
        for question in tqdm(questions, desc="Phase A", unit="q"):
            result = self.process_question(question)
            output_questions.append(self._to_bioasq_dict(result))

        output = {"questions": output_questions}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        doc_counts = [len(q['documents']) for q in output_questions]
        log.info(f"--->>>>Submission saved: {output_path}")
        log.info(f"--->>>>Avg docs: {sum(doc_counts)/len(doc_counts):.1f}  min={min(doc_counts)}  max={max(doc_counts)}")
        log.info(f"--->>>>Questions: {len(output_questions)}")
        log.info(f"--->>>>Snippets: {sum(len(q['snippets']) for q in output_questions)}")

    @staticmethod
    def _to_bioasq_dict(result: PhaseAResult) -> dict:
        """Converts a PhaseAResult to the exact BioASQ Phase A JSON schema."""
        return { "id": result.question_id, 
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
    p = argparse.ArgumentParser(description="BioASQ Task 14b Phase A submission pipeline.")
    
    p.add_argument("--input", required=True, help="Path to the BioASQ batch JSON (questions only)")
    p.add_argument("--output", required=True, help="Path for the Phase A output JSON")
    p.add_argument("--faiss-index", required=True, help="Path to the FAISS index file (pubmed_ivfsq8.index)")
    p.add_argument("--corpus", required=True, help="Path to the PubMed JSONL corpus file or directory")
    p.add_argument("--pisa-index", required=True, help="Path to the PISA no-stemmer index directory")
    p.add_argument("--top-k-retrieval", type=int, default=1000, help="Top-K per retriever before fusion (default: 1000)")
    p.add_argument("--top-k-docs", type=int, default=10,  help="Final top-N documents per question (max 10 per BioASQ rules)")
    p.add_argument("--top-snippets", type=int, default=10, help="Max snippets per question (default: 10)")
    p.add_argument("--reranker-pool", type=int, default=RERANKER_DOCS_POOL, help=f"Fusion candidates fed to cross-encoder (default: {RERANKER_DOCS_POOL})")
    p.add_argument("--ce-threshold", type=float, default=CE_THRESHOLD, help=f"CE score threshold for doc inclusion (default: {CE_THRESHOLD}). Set to -999 to disable (returns pure top-N).")
    p.add_argument("--ce-min-docs", type=int, default=CE_MIN_DOCS, help=f"Min docs always returned per question (default: {CE_MIN_DOCS})")
    p.add_argument("--debug", action="store_true", help="Enable debug logging")
    return p.parse_args()

def main():
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.top_k_docs > 10:
        log.info(f"--top-k-docs={args.top_k_docs}: returning top-{args.top_k_docs} per system.")

    pipeline = BioASQPhaseAPipeline(
        faiss_index_path = args.faiss_index,
        corpus_path = args.corpus,
        pisa_index_path = args.pisa_index,
        top_k_retrieval = args.top_k_retrieval,
        top_k_docs = args.top_k_docs,
        top_snippets = args.top_snippets,
        reranker_pool = args.reranker_pool,
        ce_threshold = args.ce_threshold if args.ce_threshold > -999 else None,
        ce_min_docs = args.ce_min_docs
    )

    pipeline.run(input_path= args.input, output_path= args.output)

if __name__ == "__main__":
    main()