import argparse
import random
from pathlib import Path

from pydantic import ValidationError

from src.core.evaluation.golden import GoldItem, load_golden, original_name
from src.models.schemas import Chunk

MIN_CHARS = 300


def load_chunks(processed_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(processed_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                chunk = Chunk.model_validate_json(line)
                if len(chunk.text) >= MIN_CHARS:
                    chunks.append(chunk)
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default="eval/golden_set.jsonl")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = load_golden(out) if out.exists() else []
    used = {(g.doc, g.page, g.start_char) for g in existing}
    used_ids = {g.id for g in existing}
    number = len(existing) + 1

    chunks = load_chunks(Path(args.data_dir) / "processed" / args.project)
    random.shuffle(chunks)
    print(f"{len(chunks)} chunks available, {len(existing)} questions so far.")

    for chunk in chunks:
        doc = original_name(chunk.metadata.get("source", ""))
        page = chunk.metadata.get("page")
        if (doc, page, chunk.start_char) in used:
            continue

        print("\n" + "=" * 70)
        print(f"{doc}   page={page}   questions so far: {len(used_ids)}")
        print("-" * 70)
        print(chunk.text)
        print("-" * 70)
        question = input("Question (Enter = skip, q = quit): ").strip()
        if question.lower() == "q":
            break
        if not question:
            continue

        while f"q{number:03d}" in used_ids:
            number += 1
        try:
            item = GoldItem(
                id=f"q{number:03d}",
                question=question,
                doc=doc,
                page=page,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                preview=chunk.text[:80].replace("\n", " "),
            )
        except ValidationError:
            print("Too short, skipped.")
            continue

        with out.open("a", encoding="utf-8") as f:
            f.write(item.model_dump_json() + "\n")
        used.add((doc, page, chunk.start_char))
        used_ids.add(item.id)

    print(f"\nDone. {len(used_ids)} questions in {out}")


if __name__ == "__main__":
    main()
