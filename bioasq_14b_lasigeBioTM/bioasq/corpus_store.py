#coprus store (lmdb)

import lmdb as _lmdb
from pathlib import Path
import json as _json
from helpers import Document
from typing import Optional
import logging

log = logging.getLogger(__name__)

class CorpusStore:
    """
    On-demand corpus lookup via LMDB.
    Pass --corpus /path/to/pubmed.lmdb when running your pipeline.
    """
 
    def __init__(self, corpus_path: str):
 
        path = Path(corpus_path)
        if not path.exists():
            raise FileNotFoundError(f"Corpus not found: {corpus_path}")
 
        log.info(f"Opening LMDB: {corpus_path} ...")
        self._env = _lmdb.open(
            str(path),
            readonly = True,
            lock = False,
            readahead = False,   
            meminit = False,
            max_readers= 256)

        with self._env.begin() as txn:
            n = txn.stat()["entries"]
        log.info(f"LMDB open: {n:,} docs")
 
    def get(self, pmid: str) -> Optional[Document]:
        
        key = str(pmid).encode()
        with self._env.begin() as txn:
            val = txn.get(key)
        if val is None:
            return None
        obj = _json.loads(val)
        return Document(
            pmid = str(pmid),
            title = obj.get("title", ""),
            content = obj.get("content", ""))
 
    def __contains__(self, pmid: str) -> bool:
        key = str(pmid).encode()
        with self._env.begin() as txn:
            return txn.get(key) is not None
 
    def close(self):
        self._env.close()