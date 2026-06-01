"""
BioASQ Task 14b — DPRF

Pipeline:
  PISA BM25         → top-1000  ──┐
  FAISS(query)      → top-1000  ──┤ seed pool for CE
                                  ↓
  Cross-encoder reranks top-n → top-50 pseudo-relevant seeds
                                  ↓
  DPRF: encode seeds → FAISS neighbours → ~500 new docs
                                  ↓
  3-way RRF( BM25, FAISS direct, DPRF expansion ) → top-n
                                  ↓
  Cross-encoder scores all → CE threshold cutoff → final docs
                                  ↓
  Snippet extraction → cross-encoder → top-10 snippets

Example of usage:
    python bioasq_pipeline_dprf.py \
  --input .../BioASQ-task14bPhaseA-testset4 \
  --output .../batch4_submission_dprf.json \
  --faiss-index  ...pubmed2026.index \
  --corpus .../pubmed2026.lmdb \
  --pisa-index   .../pubmed2026_pisa \
  --reranker-pool 500 \
  --ce-threshold  0.91 \
  --top-k-dense   1000
"""

import re
import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import pyterrier as pt
from pyterrier_pisa import PisaIndex
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

FAISS_NPROBE = 512

CROSS_ENCODER_MODEL = ""

DPRF_PSEUDO_RELEVANT_K = 50    # top docs to get from faiss in expansion
DPRF_NEIGHBOURS_PER_DOC = 10    # n of docs faiss gets per seed doc
RERANKER_DOCS_POOL = 500

RRF_K = 60

CE_THRESHOLD = 0.91   
CE_MIN_DOCS = 1 


from bioasq.helpers import Document, Snippet, PhaseAResult
from bioasq.corpus_store import CorpusStore
from bioasq.snippet_extractor import SnippetExtractor
from bioasq.sparse_retriever import SparseRetriever
from bioasq.thresholds import apply_threshold_cutoff

class DPRFExpander:
    """
    Dense Pseudo-Relevance Feedback expansion using FAISS.
    Given a list of pseudo-relevant PMIDs (seeds), encodes each seed's title+content with PubMedBERT and searches FAISS for k nearest neighbours.
    Returns {pmid_str: similarity_score} for expansion docs only, excluding any PMIDs already in the original BM25 run.
    If a doc appears as neighbour of multiple seeds, keeps the best score.
    """

    def __init__(self, index_path: str, model: SentenceTransformer,
                 corpus: "CorpusStore"):
        log.info(f"loading FAISS index from {index_path} ...")
        self.index  = faiss.read_index(index_path)
        self.index.nprobe = FAISS_NPROBE
        self.model  = model
        self.corpus = corpus
        log.info(
            f"FAISS index loaded: {self.index.ntotal:,} vectors, "
            f"nprobe={self.index.nprobe}"
        )

    def expand(self, seed_pmids: list[str], exclude_pmids: set[str], k: int = DPRF_NEIGHBOURS_PER_DOC) -> dict[str, float]:
        
        expansion: dict[str, float] = {}

        for pmid_str in seed_pmids:
            doc = self.corpus.get(pmid_str)
            if doc is None:
                log.debug(f"Seed PMID {pmid_str} not in corpus — skipping.")
                continue

            # Encode title + content as the seed query vector
            seed_text = f"{doc.title} {doc.content}".strip()
            if not seed_text:
                continue

            vec = self.model.encode(
                [seed_text], normalize_embeddings=True, show_progress_bar=False
            ).astype("float32")

            scores, pmid_ints = self.index.search(vec, k + 1)  # +1 to skip self

            for score, neighbour_int in zip(scores[0], pmid_ints[0]):
                if neighbour_int < 0:
                    continue
                neighbour_str = str(neighbour_int)
                if neighbour_str in exclude_pmids:
                    continue
                # Keep best score if seen from multiple seeds
                if neighbour_str not in expansion or score > expansion[neighbour_str]:
                    expansion[neighbour_str] = float(score)

        return expansion

    def retrieve_by_query(self, query: str, top_k: int = 1000) -> dict[str, float]:
        
        vec = self.model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")
        scores, pmid_ints = self.index.search(vec, top_k)
        return {
            str(pmid_int): float(score)
            for score, pmid_int in zip(scores[0], pmid_ints[0])
            if pmid_int >= 0
        }

class RRFFuser:
    """
    RRF via ranx.
    3-way RRF: BM25 + FAISS dense query retrieval + DPRF expansion.
    Returns sorted (pmid, rrf_score) list, best first, length ≤ top_n.
    """

    def __init__(self, k: int = RRF_K):
        self.k = k

    def combine(
        self,
        query_id: str,
        bm25_scores: dict[str, float],
        expansion_scores: dict[str, float],
        top_n: int = 300,
        dense_scores: dict[str, float] | None = None
    ) -> list[tuple[str, float]]:
    
        runs = []
        if bm25_scores:
            runs.append(Run({query_id: bm25_scores}, name="bm25"))
        if dense_scores:
            runs.append(Run({query_id: dense_scores}, name="dense"))
        if expansion_scores:
            runs.append(Run({query_id: expansion_scores}, name="dprf"))

        if not runs:
            return []
        if len(runs) == 1:
            ranked = sorted(runs[0].run[query_id].items(), key=lambda x: x[1], reverse=True)
            return ranked[:top_n]

        fused_run = fuse(
            runs = runs,
            method = "rrf",
            params = {"k": self.k},
        )

        ranked = sorted(
            fused_run.run[query_id].items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_n]


class CrossEncoderReranker:
    """Reranks documents and snippets using a cross-encoder."""

    def __init__(self, model_name: str = CROSS_ENCODER_MODEL):
        log.info(f"loading {model_name} ...")
        self.model = CrossEncoder(model_name, trust_remote_code=True)
        log.info("model ready.")

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

        return result[:top_n]

    def rerank_snippets(self, query: str, snippets: list[Snippet], top_n: int = 10) -> list[Snippet]:
        if not snippets:
            return []
        pairs  = [(query, s.text) for s in snippets]
        scores = self.model.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(snippets, scores), key=lambda x: x[1], reverse=True)
        return [s for s, _ in ranked[:top_n]]

# main

class BioASQDPRFPipeline:
    """
    Dense Pseudo-Relevance Feedback pipeline.

    Step 1 : BM25 retrieves top-1000 candidates
    Step 2 : Cross-encoder reranks → top-50 pseudo-relevant seeds
    Step 3 : For each seed, reconstruct its FAISS vector and find 10 neighbours
    Step 4 : RRF → top-100 fused pool
    Step 5 : Cross-encoder reranks fused top-100 → final top-10 documents
    Step 6 : Snippet extraction from fused top-100 pool
    Step 7 : Cross-encoder reranks snippets → top-10 snippets
    """

    def __init__(
        self,
        faiss_index_path: str,
        corpus_path: str,
        pisa_index_path: str,
        top_k_retrieval: int = 1000,
        top_k_docs:int = 10,
        top_snippets:int = 10,
        reranker_pool:int = RERANKER_DOCS_POOL,
        dprf_seeds:int = DPRF_PSEUDO_RELEVANT_K,
        dprf_neighbours:int = DPRF_NEIGHBOURS_PER_DOC,
        ce_threshold:float = CE_THRESHOLD,
        ce_min_docs:int = CE_MIN_DOCS,
        top_k_dense:int = 1000
    ):
        self.corpus = CorpusStore(corpus_path)
        self.sparse = SparseRetriever(
            index_path = pisa_index_path,
            num_results = top_k_retrieval)

        bi_encoder = SentenceTransformer(EMBEDDING_MODEL)
        self.snippet_extractor = SnippetExtractor(bi_encoder)

        self.expander = DPRFExpander(faiss_index_path, model=bi_encoder, corpus=self.corpus)

        self.reranker = CrossEncoderReranker()
        self.fuser = RRFFuser()

        self.top_k_retrieval = top_k_retrieval
        self.top_k_docs = top_k_docs
        self.top_snippets = top_snippets
        self.reranker_pool = reranker_pool
        self.dprf_seeds = dprf_seeds
        self.dprf_neighbours = dprf_neighbours
        self.ce_threshold = ce_threshold
        self.ce_min_docs = ce_min_docs
        self.top_k_dense = top_k_dense

    def process_question(self, question: dict) -> PhaseAResult:
        qtype = question.get("type", "")
        qid = question["id"]
        qbody = question["body"]

        log.debug(f"Processing [{qid}]: {qbody[:80]}...")

        bm25_scores = self.sparse.retrieve(qbody, question_type=qtype)

        if not bm25_scores:
            log.warning(f"[{qid}] BM25 returned 0 results.")
            return PhaseAResult(question_id=qid, question_type=qtype, documents=[], snippets=[])

        dense_scores = self.expander.retrieve_by_query(qbody, top_k=self.top_k_dense)
        log.debug(f"  [{qid}] FAISS direct: {len(dense_scores)} docs retrieved.")

        combined_for_seeds: dict[str, float] = {**dense_scores, **bm25_scores}
        seed_pool = sorted(
            combined_for_seeds.items(), key=lambda x: x[1], reverse=True
        )[:self.reranker_pool]

        reranked_seeds = self.reranker.rerank_documents(
            query = qbody,
            candidates = seed_pool,
            corpus = self.corpus,
            top_n = self.dprf_seeds
        )
        seed_pmids = [pmid for pmid, _ in reranked_seeds]
        log.debug(f"  [{qid}] {len(seed_pmids)} pseudo-relevant seeds selected.")

        all_retrieved = set(bm25_scores.keys()) | set(dense_scores.keys())
        expansion_scores = self.expander.expand(
            seed_pmids = seed_pmids,
            exclude_pmids = all_retrieved,
            k = self.dprf_neighbours
        )
        log.debug(f"  [{qid}] DPRF expansion: {len(expansion_scores)} new docs.")

        if not expansion_scores and not dense_scores:
            log.warning(f"[{qid}] No expansion/dense — using BM25 top-{self.reranker_pool}.")
            fused_pool = seed_pool
        else:
            fused_pool = self.fuser.combine(
                query_id = qid,
                bm25_scores = bm25_scores,
                dense_scores = dense_scores,
                expansion_scores = expansion_scores,
                top_n = self.reranker_pool
            )


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
        log.debug(f"[{qid}] Threshold {self.ce_threshold}: {len(reranked_docs)} docs selected.")

        doc_urls = [PUBMED_URL_TEMPLATE.format(pmid=pmid) for pmid, _ in reranked_docs]

        all_snippets: list[Snippet] = []
        for pmid, _ in fused_pool:
            doc = self.corpus.get(pmid)
            if doc is None:
                continue
            all_snippets.extend(
                self.snippet_extractor.extract(query=qbody, doc=doc, top_n=20)
            )

        sorted_snippets = self.reranker.rerank_snippets(
            query = qbody,
            snippets = all_snippets,
            top_n = self.top_snippets
        )

        return PhaseAResult(
            question_id = qid,
            question_type = qtype,
            documents = doc_urls,
            snippets = sorted_snippets
        )

    def run(self, input_path: str, output_path: str):
        with open(input_path, encoding="utf-8") as f:
            data = json.load(f)

        questions = data.get("questions", [])
        log.info(f"Nº de perguntas: {len(questions)} de {input_path} ...")

        output_questions = []
        for question in tqdm(questions, desc="Phase A (DPRF)", unit="q"):
            result = self.process_question(question)
            output_questions.append(self._to_bioasq_dict(result))

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"questions": output_questions}, f, indent=2, ensure_ascii=False)

        doc_counts = [len(q['documents']) for q in output_questions]
        log.info(f"--->>>> Submission saved: {output_path}")
        log.info(f"--->>>> Avg docs : {sum(doc_counts)/len(doc_counts):.1f}  "
                 f"min={min(doc_counts)}  max={max(doc_counts)}")
        log.info(f"--->>>> Questions : {len(output_questions)}")
        log.info(f"--->>>> Snippets  : {sum(len(q['snippets']) for q in output_questions)}")

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
        description="BioASQ Task 14b Phase A — DPRF pipeline."
    )
    p.add_argument("--input", required=True, help="BioASQ batch JSON (questions only)")
    p.add_argument("--output", required=True, help="Output submission JSON")
    p.add_argument("--faiss-index", required=True, help="FAISS index (pubmed_ivfsq8.index)")
    p.add_argument("--corpus", required=True,  help="PubMed JSONL corpus dir or file")
    p.add_argument("--pisa-index", required=True,  help="PISA no-stemmer index directory")
    p.add_argument("--top-k-retrieval", type=int, default=1000, help="BM25 top-K (default: 1000)")
    p.add_argument("--top-k-docs", type=int, default=10, help="Docs returned by cross-encoder per system (use 30/50/100 for ensemble, 10 for direct submission)")
    p.add_argument("--top-snippets", type=int, default=10, help="Final top-N snippets")
    p.add_argument("--reranker-pool", type=int, default=RERANKER_DOCS_POOL, help=f"Cross-encoder candidate pool (default: {RERANKER_DOCS_POOL})")
    p.add_argument("--dprf-seeds", type=int, default=DPRF_PSEUDO_RELEVANT_K,  help=f"Pseudo-relevant seeds for expansion (default: {DPRF_PSEUDO_RELEVANT_K})")
    p.add_argument("--dprf-neighbours", type=int, default=DPRF_NEIGHBOURS_PER_DOC, help=f"FAISS neighbours per seed (default: {DPRF_NEIGHBOURS_PER_DOC})")
    p.add_argument("--ce-threshold", type=float, default=CE_THRESHOLD, help=f"CE score threshold for doc inclusion (default: {CE_THRESHOLD}).")
    p.add_argument("--ce-min-docs", type=int, default=CE_MIN_DOCS, help=f"Min docs always returned per question (default: {CE_MIN_DOCS})")
    p.add_argument("--top-k-dense", type=int, default=1000, help="FAISS direct query retrieval depth (default: 1000)")
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

    pipeline = BioASQDPRFPipeline(
        faiss_index_path = args.faiss_index,
        corpus_path = args.corpus,
        pisa_index_path = args.pisa_index,
        top_k_retrieval = args.top_k_retrieval,
        top_k_docs = args.top_k_docs,
        top_snippets = args.top_snippets,
        reranker_pool = args.reranker_pool,
        dprf_seeds = args.dprf_seeds,
        dprf_neighbours = args.dprf_neighbours,
        ce_threshold = args.ce_threshold,
        ce_min_docs = args.ce_min_docs,
        top_k_dense = args.top_k_dense
    )

    pipeline.run(input_path=args.input, output_path=args.output)

if __name__ == "__main__":
    main()