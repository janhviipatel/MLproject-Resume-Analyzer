"""Extract plain text from PDF, DOCX or TXT uploads."""
import io
import re

MAX_BYTES = 5 * 1024 * 1024
BULLET = re.compile(r"^\s*[•●▪■◦·*–—-]\s+")


def extract_text(filename: str, data: bytes) -> str:
    if len(data) > MAX_BYTES:
        raise ValueError("File is larger than 5 MB.")
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        import pdfplumber
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            text = "\n".join((p.extract_text() or "") for p in pdf.pages)
    elif name.endswith(".docx"):
        import docx
        d = docx.Document(io.BytesIO(data))
        text = "\n".join(p.text for p in d.paragraphs)
    elif name.endswith((".txt", ".md")):
        text = data.decode("utf-8", errors="ignore")
    else:
        raise ValueError("Unsupported file type. Upload a PDF, DOCX or TXT file.")
    if not text.strip():
        raise ValueError("No text found. Scanned PDFs need OCR before upload.")
    return normalize(text)


def normalize(text: str) -> str:
    """Unify bullet glyphs to '- ' and tidy whitespace."""
    lines = []
    for line in text.replace("\r", "").split("\n"):
        line = line.replace("\u00a0", " ").rstrip()
        lines.append(BULLET.sub("- ", line) if BULLET.match(line) else line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
