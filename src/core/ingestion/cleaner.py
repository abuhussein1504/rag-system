import re
import unicodedata

_PRIVATE_USE = re.compile(r"[\ue000-\uf8ff]")

def clean_text(text: str, *, join_lines: bool = True) -> str:
    # Fix odd characters, e.g. the ligature "ﬁ" becomes "fi"
    text = unicodedata.normalize("NFKC", text)
    text = _PRIVATE_USE.sub("-", text)

    # Join words split across lines: "inter-\nnational" -> "international"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Many spaces/tabs -> one space, and no spaces around line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)

    # A single newline inside a paragraph -> space (PDFs break lines randomly)
    if join_lines:
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Max one blank line between paragraphs
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

