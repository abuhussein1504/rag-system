import math
import re
from collections import Counter

from src.models.enums import RetrievalSource
from src.models.schemas import Chunk, RetrievedChunk

_WORD = re.compile(r"\w+")
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the to was were "
    "will with".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        docs = [tokenize(text) for text in texts]
        self.doc_len = [len(tokens) for tokens in docs]
        self.avg_len = sum(self.doc_len) / len(docs) if docs else 0.0

        self.postings: dict[str, list[tuple[int, int]]] = {}
        for doc_id, tokens in enumerate(docs):
            for term, tf in Counter(tokens).items():
                self.postings.setdefault(term, []).append((doc_id, tf))

        n = len(docs)
        self.idf = {
            term: math.log(1 + (n - len(posting) + 0.5) / (len(posting) + 0.5))
            for term, posting in self.postings.items()
        }

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        scores: dict[int, float] = {}
        for term in set(tokenize(query)):
            for doc_id, tf in self.postings.get(term, []):
                norm = 1 - self.b + self.b * self.doc_len[doc_id] / self.avg_len
                scores[doc_id] = scores.get(doc_id, 0.0) + self.idf[term] * (
                    tf * (self.k1 + 1) / (tf + self.k1 * norm)
                )
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:k]


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.index = BM25Index([c.text for c in chunks], k1, b)

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(chunk=self.chunks[i], score=score, source=RetrievalSource.SPARSE)
            for i, score in self.index.search(query, k)
        ]

