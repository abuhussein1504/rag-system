import argparse
import random
from pathlib import Path

from pydantic import ValidationError

from src.core.evaluation.golden import GoldItem, load_golden, original_name
from src.models.schemas import Chunk

MIN_CHARS = 300

PROJECT_1_QUESTIONS: dict[tuple[str, int | None, int], str] = {
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 2, 0): "What does Abdullah Hussein declare regarding plagiarism in the report?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 3, 0): "What is Make My Money Last (3ML) and what key metrics does it calculate according to the abstract?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 4, 0): "Which Quranic verse and supervisor does Abdullah Hussein acknowledge in his thesis acknowledgments?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 5, 0): "What sections are included under Chapter 1 Introduction in the table of contents?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 5, 768): "What introductory sections covering deliverables, scope, and project planning precede Chapter 2 in the contents?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 5, 1529): "Which theoretical background and existing solutions are listed in Chapter 2 of the table of contents?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 5, 2244): "Which competitors and comparative discussion sections are listed in the literature review contents?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 5, 3024): "What requirement categories are outlined under Chapter 3 in the table of contents?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 6, 0): "Which system modelling diagrams are listed under section 3.4 in the table of contents?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 6, 830): "What testing and evaluation subsections are outlined at the end of Chapter 3 in the contents?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 6, 1660): "Which sections detailing system architecture and Model 1 Classifier appear in Chapter 4 contents?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 6, 2486): "Where are the design and training configurations of Model 2 Event Parser and Model 3 Coach listed in the contents?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 6, 3279): "What subsections describe the dataset construction and training configuration for Model 3 in the table of contents?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 7, 0): "Which mobile application implementation and budget engine sections are listed in the Chapter 4 contents?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 7, 784): "What testing methodologies and verification sections are listed under section 4.10 in the contents?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 7, 1599): "Which evaluation sections for the AI models are listed in Chapter 5 in the table of contents?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 7, 2426): "Where are application functional evaluation and skills acquired listed in the table of contents?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 7, 3267): "What concluding sections covering contributions, lessons learned, and future work are listed in Chapter 6 contents?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 9, 0): "Which figures illustrate the business model, project schedule, and competitor apps in the list of figures?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 9, 777): "Which system design diagrams including use case, DFD, and class diagrams are indexed in the list of figures?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 9, 1628): "Which figures display the training results and model configuration for the Event Parser in the list of figures?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 9, 2429): "Which figures in the list of figures index the Coach model training loss and mobile application screens?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 9, 3256): "Which figures show the budget chat interface, analytics donut chart, and confusion matrix in the list of figures?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 11, 0): "Why do households need a reliable daily spending reference rather than another spreadsheet?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 11, 802): "How does the app address complexity overload and daily variable expenses in its problem definition?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 12, 0): "What are the primary aims and objectives of the 3ML project regarding daily budgeting?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 12, 798): "What guidelines govern micro-coaching and privacy by default in 3ML's objectives?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 13, 0): "What backend services and proposed architectural solutions are specified for the 3ML scope?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 16, 0): "How does 3ML leverage mental accounting and dual-process theory to help users decide if they can spend today?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 16, 795): "Why does 3ML implement a Just-in-Time Adaptive Intervention (JITAI) loop for daily spending feedback?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 17, 0): "What four design imperatives synthesize the theoretical foundations of the 3ML application?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 17, 800): "Why is the budgeting logic implemented as a deterministic rules engine rather than an LLM in 3ML?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 18, 0): "How is the Design Science Research (DSR) methodology applied in developing 3ML, and how does PocketGuard calculate its allowance?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 18, 805): "What are the limitations of PocketGuard's bank account aggregation approach compared to 3ML?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 19, 0): "How do Quicken Simplifi's Spending Plan and Revolut's budgeting features compare in tracking left-to-spend amounts?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 20, 0): "How does 3ML compare to PocketGuard, Quicken Simplifi, and Revolut in terms of input method and core concept?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 20, 790): "How does 3ML ensure formula transparency compared to competitors using its Tap-to-Justify feature?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 21, 0): "Why does 3ML mandate a savings-first logic in calculating the daily allowance?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 21, 801): "What is the Runway Buffer in 3ML and how does it reward users for underspending?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 21, 1599): "What is the Quick-Split Income Prompt in 3ML and how does it assist users when logging salary?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 22, 0): "What key shortcomings of existing personal finance tools are summarized at the conclusion of Chapter 2?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 22, 801): "What foundational principles justify 3ML's focus on a single Safe-to-Spend number and Runway to Payday?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 24, 0): "How does Chapter 3 translate 3ML's high-level aims into specific functional and non-functional requirements?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 24, 795): "What functional requirements govern natural-language transaction logging and micro-coaching generation in 3ML?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 25, 0): "What non-functional requirements for privacy, offline functionality, and latency are specified for 3ML?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 25, 793): "What are the four core components of the 3ML system architecture and their respective responsibilities?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 26, 0): "What interactions are captured in 3ML's use case diagram when a user logs a transaction?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 27, 0): "How does data flow from user text entry through the Event Parser, local store, and Rules Engine in 3ML's DFD?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 28, 0): "What core entities and relationships are modeled in 3ML's class diagram?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 29, 0): "How does the sequence diagram illustrate parallel execution of budget calculation and coaching tip generation?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 31, 0): "What confidence check does the NLP flow perform to decide whether to prompt the user with a clarifying question?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 31, 803): "How does the system handle over-budget versus under-budget states when generating the coaching tip and daily limit?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 32, 0): "Why did 3ML choose a Two-Model Split with deterministic rules instead of an end-to-end Pure LLM approach?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 32, 799): "What three testing layers are planned in section 3.6 to evaluate 3ML's mathematical correctness and integration?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 32, 1598): "What evaluation metrics are used to measure the parsing accuracy of 3ML's NLP models?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 33, 0): "What latency goal and financial solvency metric are defined for evaluating 3ML's user experience?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 33, 802): "What sample sizes and test datasets are designated for validating the AI parser and user experience?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 33, 1600): "How does the Chapter 3 summary justify the architectural decisions made for 3ML?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 35, 0): "What major implementation topics and system components are introduced in Chapter 4?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 35, 797): "How does 3ML's system architecture separate client-side logic from remote AI backend services?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 35, 1594): "How does AppProvider.sendMessage implement a parser-first message handling flow in the Flutter application?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 36, 0): "What is the custom dataset abuhussein1504/event-parser-data created for this project and where is it hosted?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 36, 794): "How were diverse Kaggle financial datasets unified into 3ML's sixteen target categories and augmented for robustness?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 37, 0): "What is the architecture and primary function of Model 1: The Classifier in the 3ML pipeline?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 37, 804): "What dataset composition and training configuration were used to fine-tune the DistilBERT intent classifier?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 38, 0): "What test accuracy and F1-score did the DistilBERT intent classifier achieve on held-out evaluation data?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 38, 800): "Which base language model and quantization technique were chosen for Model 2: Event Parser?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 39, 0): "What QLoRA hyperparameters and target modules were configured for fine-tuning the Qwen2.5-3B Event Parser?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 41, 0): "What exact JSON schema fields are produced by Model 2: Event Parser when extracting transaction events?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 41, 800): "What base model and dataset sources were used to develop Model 3: Coach and Chatbot?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 42, 0): "What training hyperparameter settings were used for fine-tuning the Llama-3.2-3B coach model?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 44, 0): "How did Model 3 respond in sample tests for coach mode and chatbot mode under over-budget conditions?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 45, 0): "What core technology stack and state management libraries power the 3ML mobile application?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 45, 805): "What five architectural layers structure the Flutter mobile client code in 3ML?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 46, 0): "What visual elements and budget metrics are featured on 3ML's Home Dashboard screen?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 47, 0): "What information and formula steps are shown in the Tap-to-Justify calculation breakdown modal?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 49, 0): "How does the Budget Chat interface present user transaction logs, system confirmations, and coaching tips?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 50, 0): "What visualizations are provided in the Spending Breakdown screen to analyze category expenses?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 52, 0): "What configuration options and financial summary details are displayed in the Settings tab?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 54, 0): "What user privacy guarantees are confirmed in 3ML's Privacy Policy Modal?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 55, 0): "What profile parameters can be configured and updated on the Edit Profile Screen?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 56, 0): "What exact mathematical formulas are implemented in BudgetService to compute daily Safe-to-Spend?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 56, 798): "How does BudgetService calculate daily rollover credits for the Runway Buffer mechanism?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 56, 1602): "How does 3ML persist user settings and allow data export into CSV and JSON formats?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 57, 0): "What challenges regarding dataset inconsistency and offline resilience were encountered during development?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 57, 794): "How was white-box testing conducted across the budget arithmetic and AppProvider message flow?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 57, 1590): "What specific logic units were validated through unit testing in BudgetService and DateParserService?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 58, 0): "How did system testing verify 3ML's end-to-end user workflows and offline fallback behavior?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 58, 800): "What test cases and expected outcomes demonstrated full system compliance during offline and profile update tests?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 59, 0): "How did acceptance testing verify that all functional requirements from Chapter 3 were met?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 59, 803): "How were verification and validation performed to confirm that 3ML effectively solves users' financial tracking problems?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 59, 1602): "What key artifacts and models were successfully developed and integrated by the end of Chapter 4?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 61, 0): "How was the DistilBERT intent classifier evaluated in Chapter 5, and what test set was used?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 61, 800): "What evaluation results and JSON structural validity were achieved by the Qwen2.5-3B Event Parser?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 61, 1599): "How does the parsing system handle ambiguous inputs and resolve natural-language date expressions into DateTime objects?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 62, 0): "How did the fine-tuned Llama-3.2-3B coach model perform when generating financial advice for deficit versus surplus scenarios?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 62, 798): "How was the mobile application evaluated under extreme financial edge cases such as zero income or high fixed bills?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 62, 1592): "What benefits were realized from decoupling probabilistic language models from the deterministic budgeting engine in 3ML?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 63, 0): "What machine learning and software engineering skills were acquired during the development of 3ML?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 63, 801): "What professional and systems integration disciplines were reinforced during the development lifecycle of 3ML?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 65, 0): "What is the comprehensive project summary of Make My Money Last (3ML) presented in Chapter 6?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 65, 802): "How does Chapter 6 evaluate the successful achievement of 3ML's primary and secondary objectives?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 65, 1601): "What privacy-by-default and offline design objectives were validated in the final project evaluation?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 66, 0): "What three principal technical and data contributions were delivered by the 3ML project?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 66, 801): "What key lesson was learned regarding model size versus domain-specific fine-tuning during 3ML's development?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 66, 1604): "What future enhancements such as recurring transaction detection and on-device SLMs are proposed for 3ML?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 67, 0): "What final conclusions does the author draw regarding the viability of domain-tuned small models for personal financial management?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 68, 0): "Which economic and competitor references are cited in entries 1 through 5 of the reference list?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 68, 825): "Which academic works on mental accounting, dual-process theory, and the author's event-parser-data dataset are cited in references 6 to 11?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 69, 0): "Which open-source datasets and pre-trained models including DistilBERT and Qwen2.5 are cited in references 12 to 16?",
    ("Abdullah_Hussein_Mohammed_-_22513175_-_FULL.pdf", 69, 793): "Which machine learning frameworks and mobile libraries including Unsloth and Llama-3.2 are cited in references 17 to 20?",
}


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
    parser.add_argument("--shuffle", action="store_true", default=False)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = load_golden(out) if out.exists() else []
    used = {(g.doc, g.page, g.start_char) for g in existing}
    used_ids = {g.id for g in existing}
    number = len(existing) + 1

    chunks = load_chunks(Path(args.data_dir) / "processed" / args.project)
    if args.shuffle:
        random.shuffle(chunks)
    else:
        chunks.sort(key=lambda c: (c.metadata.get("page") or 0, c.start_char))

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

        default_q = PROJECT_1_QUESTIONS.get((doc, page, chunk.start_char), "")
        prompt_str = "Question (Enter = skip, q = quit): "

        try:
            raw_input = input(prompt_str).strip()
            if raw_input.lower() == "q":
                break
            question = raw_input if raw_input else default_q
        except EOFError:
            if default_q:
                print(f"{prompt_str}{default_q}")
                question = default_q
            else:
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
                answers=[chunk.text],
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
