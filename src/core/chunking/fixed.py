import hashlib

from src.models.schemas import Chunk, Document


class FixedSizeChunker:
    def __init__(self, chunk_size: int = 1000, overlap: int = 200) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap must be >= 0 and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.text
        n = len(text)
        chunks: list[Chunk] = []
        start = 0

        while start < n:
            end = self._find_end(text, start, n)

            raw = text[start:end]
            piece = raw.strip()
            if piece:
                # Offsets point to the stripped text inside the original document
                s = start + (len(raw) - len(raw.lstrip()))
                e = s + len(piece)
                chunks.append(
                    Chunk(
                        id=self._make_id(doc.id, s, e, piece),
                        doc_id=doc.id,
                        text=piece,
                        start_char=s,
                        end_char=e,
                        metadata={
                            **doc.metadata,
                            "source": doc.source,
                            "chunk_index": len(chunks),
                        },
                    )
                )

            if end >= n:
                break
            start = self._next_start(text, start, end)

        return chunks

    def _find_end(self, text: str, start: int, n: int) -> int:
        end = min(start + self.chunk_size, n)
        if end < n:
            # Cut at the last whitespace in the final 20% of the window
            window_start = start + int(self.chunk_size * 0.8)
            cut = max(
                text.rfind(" ", window_start, end),
                text.rfind("\n", window_start, end),
            )
            if cut > start:
                end = cut
        return end

    def _next_start(self, text: str, start: int, end: int) -> int:
        nxt = max(end - self.overlap, start + 1)  # always move forward
        # Move to the beginning of a word, so a chunk doesn't start mid-word
        while nxt < end and not text[nxt - 1].isspace():
            nxt += 1
        return nxt

    @staticmethod
    def _make_id(doc_id: str, start: int, end: int, text: str) -> str:
        raw = f"{doc_id}:{start}:{end}:{text}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
