"""
tests/phase8/test_ingestion.py

Phase 8 — Document Ingestion Tests
Covers:
  - .txt and .pdf loading from directory
  - Text cleaning and normalization
  - Metadata tagging (document_name, page, file_type, source)
  - Empty / missing directory handling
  - PDF page-level splitting
"""
import os
import shutil
import tempfile

import pytest
from langchain_core.documents import Document

from src.rag_pipeline import clean_and_normalize_text, load_documents_from_directory


# ---------------------------------------------------------------------------
# Text Cleaning
# ---------------------------------------------------------------------------

class TestTextCleaning:

    def test_removes_null_bytes(self):
        raw = "Hello\x00World"
        assert "\x00" not in clean_and_normalize_text(raw)

    def test_resolves_hyphenated_line_breaks(self):
        raw = "Trans-\nformer models use self-attention."
        result = clean_and_normalize_text(raw)
        assert "Transformer" in result
        assert "Trans-\nformer" not in result

    def test_collapses_multiple_blank_lines(self):
        raw = "Paragraph one.\n\n\n\n\nParagraph two."
        result = clean_and_normalize_text(raw)
        assert "\n\n\n" not in result

    def test_collapses_consecutive_spaces(self):
        raw = "Too    many   spaces   here."
        result = clean_and_normalize_text(raw)
        assert "  " not in result

    def test_normalises_non_breaking_spaces(self):
        raw = "Price\xa0is\xa0100."
        result = clean_and_normalize_text(raw)
        assert "\xa0" not in result

    def test_empty_input_returns_empty(self):
        assert clean_and_normalize_text("") == ""
        assert clean_and_normalize_text(None) == ""  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Document Ingestion from Directory
# ---------------------------------------------------------------------------

class TestDocumentIngestion:

    def setup_method(self):
        self.dir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _write(self, name, content):
        path = os.path.join(self.dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_loads_txt_file(self):
        self._write("sample.txt", "Hello AI world.")
        docs = load_documents_from_directory(self.dir)
        names = [d.metadata["document_name"] for d in docs]
        assert "sample.txt" in names

    def test_txt_metadata_fields(self):
        self._write("notes.txt", "Some content here.")
        docs = load_documents_from_directory(self.dir)
        doc = next(d for d in docs if d.metadata["document_name"] == "notes.txt")
        assert doc.metadata["page"] == 1
        assert doc.metadata["file_type"] == "txt"
        assert "notes.txt" in doc.metadata["source"]

    def test_loads_multiple_files(self):
        self._write("a.txt", "File A content.")
        self._write("b.txt", "File B content.")
        docs = load_documents_from_directory(self.dir)
        names = {d.metadata["document_name"] for d in docs}
        assert "a.txt" in names
        assert "b.txt" in names

    def test_ignores_non_pdf_txt_files(self):
        self._write("readme.md", "# Markdown")
        self._write("config.json", "{}")
        self._write("data.csv", "a,b,c")
        docs = load_documents_from_directory(self.dir)
        assert all(
            d.metadata["document_name"].endswith((".txt", ".pdf")) for d in docs
        )

    def test_empty_directory_returns_empty(self):
        docs = load_documents_from_directory(self.dir)
        assert docs == []

    def test_missing_directory_returns_empty(self):
        docs = load_documents_from_directory("/nonexistent/path/xyz")
        assert docs == []

    def test_txt_content_is_cleaned(self):
        self._write("messy.txt", "Hello\x00  World\n\n\n\nEnd.")
        docs = load_documents_from_directory(self.dir)
        content = docs[0].page_content
        assert "\x00" not in content
        assert "Hello" in content

    def test_pdf_ingestion(self, tmp_path):
        """Create a minimal 1-page PDF and verify it loads correctly."""
        from pypdf import PdfWriter
        pdf_path = str(tmp_path / "test.pdf")
        writer = PdfWriter()
        page = writer.add_blank_page(width=200, height=200)
        with open(pdf_path, "wb") as f:
            writer.write(f)
        shutil.copy(pdf_path, os.path.join(self.dir, "test.pdf"))
        # blank PDF pages produce empty text — ingestion should not crash
        docs = load_documents_from_directory(self.dir)
        # blank pages have no content and are skipped; just confirm no exception
        assert isinstance(docs, list)

