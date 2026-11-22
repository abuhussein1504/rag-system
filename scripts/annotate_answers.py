import argparse
import json
import re
from pathlib import Path

from pydantic import ValidationError

from src.core.evaluation.golden import GoldItem, original_name
from src.core.ingestion.registry import load_file


def load_pages(raw_dir: Path, project: str) -> dict[tuple[str, int], str]:
    pages: dict[tuple[str, int], str] = {}
    for path in sorted(raw_dir.iterdir()):
        if path.is_file():
            for doc in load_file(path, project):
                pages[(original_name(path.name), doc.metadata.get("page", 1))] = doc.text
    return pages


def sentences(text: str) -> list[str]:
    flat = re.sub(r"\s+", " ", text)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", flat) if s.strip()]


def parse_selection(raw: str, count: int) -> list[int] | None:
    picked: list[int] = []
    try:
        for part in raw.split(","):
            if "-" in part:
                start, end = part.split("-")
                picked.extend(range(int(start), int(end) + 1))
            else:
                picked.append(int(part))
    except ValueError:
        return None
    if not picked or any(n < 1 or n > count for n in picked):
        return None
    return sorted(set(picked))


def group_runs(numbers: list[int]) -> list[list[int]]:
    groups = [[numbers[0]]]
    for n in numbers[1:]:
        if n == groups[-1][-1] + 1:
            groups[-1].append(n)
        else:
            groups.append([n])
    return groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--src", default="eval/golden_set.jsonl")
    parser.add_argument("--out", default="eval/golden_set_v2.jsonl")
    args = parser.parse_args()

    pages = load_pages(Path(args.data_dir) / "raw" / args.project, args.project)
    old_items = [
        json.loads(line)
        for line in Path(args.src).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    out = Path(args.out)
    done: set[str] = set()
    if out.exists():
        done = {
            json.loads(line)["id"]
            for line in out.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    for old in old_items:
        if old["id"] in done:
            continue
        page = old.get("page") or 1
        while True:
            text = pages.get((old["doc"], page))
            sents = sentences(text) if text else []
            print("\n" + "=" * 70)
            print(f"{old['id']}: {old['question']}")
            print(f"{old['doc']}  page {page}   ({len(done)}/{len(old_items)} done)")
            print("-" * 70)
            for i, sentence in enumerate(sents, start=1):
                print(f"{i:>3}. {sentence[:300]}")

            raw = input(
                "\nAnswer sentences (e.g. 4 or 4-5), < prev page, > next page, "
                "s skip question, q quit: "
            ).strip().lower()

            if raw == "q":
                return
            if raw == "s":
                break
            if raw == "<":
                page = max(1, page - 1)
                continue
            if raw == ">":
                page += 1
                continue

            numbers = parse_selection(raw, len(sents))
            if numbers is None:
                print("Not understood, try again.")
                continue
            answers = [" ".join(sents[n - 1] for n in group) for group in group_runs(numbers)]
            try:
                item = GoldItem(
                    id=old["id"], question=old["question"], doc=old["doc"],
                    page=page, answers=answers,
                )
            except ValidationError as exc:
                print(f"Invalid: {exc.errors()[0]['msg']}")
                continue

            with out.open("a", encoding="utf-8") as f:
                f.write(item.model_dump_json() + "\n")
            done.add(old["id"])
            break

    print(f"\nDone: {len(done)} questions in {out}")


if __name__ == "__main__":
    main()

