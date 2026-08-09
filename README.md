# BioASQ Task 14b — Biomedical Question Answering

Participation in the [BioASQ](http://bioasq.org/) Challenge **Task 14b**, covering
both Phase A (document and snippet retrieval) and Phase B (exact and ideal answer
generation).

> Developed by Diogo Antunes under the supervision of Francisco M. Couto
> (LASIGE, Faculty of Sciences, University of Lisbon).
>
> For the Working Notes click [here](https://clef-staging.pages.dev/paper6.pdf)

---

## Overview

The system is split into two phases:

- **Phase A** retrieves relevant PubMed documents and extracts answer-bearing
  snippets. Four interchangeable pipelines were evaluated (see below).
- **Phase B** takes the Phase A output and generates the *exact* answer
  (yes/no, factoid, list) and the *ideal* answer (summary) with a few-shot
  prompted LLM.

### Phase A pipelines

| Pipeline | File | Approach |
|---|---|---|
| **Hybrid** | `pipelines/bioasq_pipeline_normal.py` | FAISS dense (PubMedBERT) + PISA BM25, weighted-sum fusion, cross-encoder reranking, score-threshold cutoff |
| **Sparse-Only** | `pipelines/bioasq_pipeline_onlyBM25.py` | PISA BM25 only → cross-encoder reranking |
| **DPRF** | `pipelines/bioasq_pipeline_dprf.py` | Dense pseudo-relevance feedback expansion, RRF fusion of BM25 + dense runs, cross-encoder reranking |
| **Ensemble** | `pipelines/Ensemble_crossencoders.py` | Reciprocal-rank fusion across multiple cross-encoders |

### Phase B

| File | Approach |
|---|---|
| `pipelines/phaseb.py` | Few-shot prompting with `google/gemma-4-E4B-it`; two-pass generation (exact answer → ideal answer) |

---

## Repository structure

```
.
├── bioasq/                       # shared modules imported by the pipelines
│   ├── helpers.py                
│   ├── corpus_store.py           
│   ├── snippet_extractor.py      
│   ├── sparse_retriever.py       
│   ├── thresholds.py             
│   └── wsum_fuser.py             
├── pipelines/                    # runnable entry points (one per system)
│   ├── bioasq_pipeline_normal.py
│   ├── bioasq_pipeline_onlyBM25.py
│   ├── bioasq_pipeline_dprf.py
│   ├── Ensemble_crossencoders.py
│   └── phaseb.py
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. create a virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. install PyTorch matching your CUDA version FIRST
#    see https://pytorch.org/get-started/locally/

# 3. install the rest
pip install -r requirements.txt
```

### Prerequisites (not included in this repo)

These are large and must be obtained separately:

**Prebuilt indexes (FAISS + PISA)** used in this work are available on Hugging Face:
> 🤗 [dantunes6/lean-rag-indexes](https://huggingface.co/datasets/dantunes6/lean-rag-indexes)

Download them with:
```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="dantunes6/lean-rag-indexes",
    repo_type="dataset",
    local_dir="./lean-rag-indexes"
)
```
---

## Usage

### Phase A — example (Hybrid pipeline)

```bash
python pipelines/bioasq_pipeline_normal.py \
    --input         path/to/BioASQ-task14bPhaseA-testset.json \
    --output        path/to/submission.json \
    --faiss-index   path/to/pubmed.index \
    --corpus        path/to/pubmed.lmdb \
    --pisa-index    path/to/pubmed_pisa \
    --top-k-retrieval 1000 \
    --top-k-docs 10 \
    --top-snippets 10 \
    --ce-threshold 0.91
```

Each pipeline exposes `--help` for its full set of arguments.

### Phase B — answer generation

```bash
python pipelines/phaseb.py
```

For the phaseb.py script the paths should be added in the file.

---

## Citation

Developed at LASIGE, University of Lisbon by Diogo Antunes.
Supervised by Francisco M. Couto.


