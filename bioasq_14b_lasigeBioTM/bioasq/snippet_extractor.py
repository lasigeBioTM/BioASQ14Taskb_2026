#Snippet extractor 

import spacy
import numpy as np
from sentence_transformers import SentenceTransformer
from helpers import Document, Snippet

class SnippetExtractor:
    """
    Sliding window snippet extraction (1/2/3-sentence windows).
    scispaCy for sentence splitting, PubMedBERT bi-encoder for pre-filtering.
    """

    def __init__(self, model: SentenceTransformer):
        self.model = model
        self.nlp   = spacy.load(
            "en_ner_bionlp13cg_md",
            disable=["ner", "tagger", "attribute_ruler", "lemmatizer"]
        )

    def _split_sentences(self, text: str) -> list[tuple[str, int, int]]:
        doc = self.nlp(text)
        return [
            (sent.text.strip(), sent.start_char, sent.end_char)
            for sent in doc.sents if sent.text.strip()
        ] 

    def _build_candidates(
        self, section_name: str, text: str
    ) -> list[tuple[str, str, int, int]]:
        sents = self._split_sentences(text)
        candidates = []
        n= len(sents)

        for i in range(n):
            _, s1_start, s1_end = sents[i]
            candidates.append((text[s1_start:s1_end], section_name, s1_start, s1_end))
            if i + 1 < n:
                _, _, s2_end = sents[i + 1]
                candidates.append((text[s1_start:s2_end], section_name, s1_start, s2_end))
            if i + 2 < n:
                _, _, s3_end = sents[i + 2]
                candidates.append((text[s1_start:s3_end], section_name, s1_start, s3_end))

        return candidates

    def extract(self, query: str, doc: Document, top_n: int = 20) -> list[Snippet]:
        candidates = []
        if doc.title:
            candidates.extend(self._build_candidates("title", doc.title))
        if doc.content:
            candidates.extend(self._build_candidates("abstract", doc.content))
        if not candidates:
            return []

        texts = [query] + [c[0] for c in candidates]
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        query_vec = embeddings[0]
        cand_vecs = embeddings[1:]
        scores = cand_vecs @ query_vec

        ranked_indices = np.argsort(scores)[::-1][:top_n]
        snippets = []
        for idx in ranked_indices:
            text, section, start, end = candidates[idx]
            snippets.append(Snippet(
                document = doc.url,
                text = text,
                begin_section = section,
                end_section = section,
                offset_in_begin = start,
                offset_in_end = end,
            ))
        return snippets