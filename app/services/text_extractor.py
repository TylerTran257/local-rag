import re
from pathlib import Path

from pypdf import PdfReader


class TextExtractionError(ValueError):
    """Raised when an uploaded file cannot be turned into text."""


class TextExtractor:
    def extract(self, file_path: Path) -> str:
        suffix = file_path.suffix.lower()
        if suffix == ".txt":
            return self._extract_txt_file(file_path)

        if suffix == ".pdf":
            return self._extract_pdf_file(file_path)

        raise TextExtractionError(f"Unsupported file type: {file_path.suffix}")

    def _extract_txt_file(self, file_path: Path) -> str:
        return file_path.read_bytes().decode("utf-8")

    def _normalize_pdf_text(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def _extract_pdf_file(self, file_path: Path) -> str:
        reader = PdfReader(str(file_path))
        page_text = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(page_text).strip()
        normalized_text = self._normalize_pdf_text(text)

        if not text:
            raise TextExtractionError("PDF did not contain extractable text")

        return normalized_text
