#helpers

from dataclasses import dataclass, field

PUBMED_URL_TEMPLATE = "http://www.ncbi.nlm.nih.gov/pubmed/{pmid}"

@dataclass
class Document:
    pmid: str
    title: str = ""
    content: str = ""      

    @property
    def url(self) -> str:
        return PUBMED_URL_TEMPLATE.format(pmid=self.pmid)

    @property
    def full_text(self)-> str:
        parts = []
        if self.title: parts.append(self.title)
        if self.content: parts.append(self.content)
        return " ".join(parts)


@dataclass
class Snippet:
    document: str         
    text: str
    begin_section: str          
    end_section: str
    offset_in_begin: int
    offset_in_end: int

    #for the ensemble pipeline
    def key(self)-> str:
        return f"{self.document}__{self.offset_in_begin}__{self.offset_in_end}"


@dataclass
class PhaseAResult:
    question_id: str
    question_type: str
    documents: list[str]         
    snippets: list[Snippet]       

@dataclass
class ModelScores:
    """For the Ensemble pipeline"""
    model_name: str
    doc_scores: dict[str, float] = field(default_factory=dict)
    snip_scores: dict[str, float] = field(default_factory=dict)
    top_docs: list[str] = field(default_factory=list)
    top_snippets: list[Snippet] = field(default_factory=list)