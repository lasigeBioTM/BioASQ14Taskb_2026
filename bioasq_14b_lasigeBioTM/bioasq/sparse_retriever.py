#Sparse retriever
import re
import logging
import pyterrier as pt
from pyterrier_pisa import PisaIndex

log = logging.getLogger(__name__)

BM25_K1 = 0.6
BM25_B = 0.4
BM25_THREADS = 50

class SparseRetriever:
    
    QUESTION_WORDS = re.compile(
        r'\b(what|which|where|when|who|how many|how much|is|are|was|were|'
        r'does|do|did|has|have|the|a|an|of|in|for|to|be|been|by|at|'
        r'indicate|indication|used|treatment|role|purpose|effect|aim|'
        r'describe|list|name|please|can|could|give|provide)\b', re.IGNORECASE)
 
    def __init__(
        self,
        index_path: str,
        num_results: int = 1000
    ):
        if not pt.java.started():
            pt.java.init()
 
        log.info(f"Loading PISA: {index_path} ...")
        pisa = PisaIndex(
            path=index_path, text_field=["title", "content"], stemmer="porter2", stops="terrier") #!!!has to mach index configurations
        self.retriever = pisa.bm25(
            k1=BM25_K1, b=BM25_B, threads=BM25_THREADS, num_results=num_results
        )
        log.info(f"SparseRetriever ready — k1={BM25_K1}, b={BM25_B}")
 
        self.num_results = num_results
 
    def expand_query(self, question_body: str, question_type: str) -> str:
        if question_type == "summary":
            return question_body
        key_terms = self.QUESTION_WORDS.sub(' ', question_body)
        key_terms = re.sub(r'[?().,;:]', ' ', key_terms)
        key_terms = re.sub(r'\s+', ' ', key_terms).strip()
        if not key_terms:
            return question_body
        if question_type == "factoid":
            return f"{question_body} {key_terms} {key_terms}"
        elif question_type in ("yesno", "list"):
            return f"{question_body} {key_terms}"
        return question_body
 
    def retrieve(self, query: str, question_type: str = "") -> dict[str, float]:
        expanded = self.expand_query(query, question_type)

        df = self.retriever.search(expanded)
        scores = {
            str(docno): float(score)
            for docno, score in zip(df["docno"], df["score"])
        }

        log.debug(f"BM25: {len(scores)} docs → top-{self.num_results}")
        return dict(
            sorted(scores.items(), key=lambda x: x[1], reverse=True)[:self.num_results]
        ) 

