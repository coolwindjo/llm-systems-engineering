from __future__ import annotations

import argparse
import re
from pathlib import Path

from pypdf import PdfReader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a PDF file into a plain Markdown text file."
    )
    parser.add_argument("pdf_path", type=Path, help="Path to the source PDF file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional output Markdown path. Defaults to the PDF path with a .md suffix.",
    )
    parser.add_argument(
        "--no-page-headings",
        action="store_true",
        help="Do not add '## Page N' headings between extracted pages.",
    )
    return parser.parse_args()


def _normalize_page_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00ad", "")
    text = re.sub(r"([a-zA-Z0-9])-\n([a-z])", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)

    normalized_lines: list[str] = []
    previous_blank = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if not previous_blank:
                normalized_lines.append("")
            previous_blank = True
            continue
        normalized_lines.append(line)
        previous_blank = False

    return "\n".join(normalized_lines).strip()


def convert_pdf_to_markdown(
    pdf_path: Path,
    output_path: Path | None = None,
    *,
    include_page_headings: bool = True,
) -> Path:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {pdf_path.name}")

    output_path = output_path or pdf_path.with_suffix(".md")
    reader = PdfReader(str(pdf_path))

    sections = [
        f"# {pdf_path.stem}",
        "",
        f"- Source PDF: `{pdf_path.name}`",
        f"- Total pages: `{len(reader.pages)}`",
        "",
    ]

    extracted_page_count = 0
    for page_index, page in enumerate(reader.pages, start=1):
        page_text = _normalize_page_text(page.extract_text() or "")
        if not page_text:
            continue

        extracted_page_count += 1
        if include_page_headings:
            sections.extend([f"## Page {page_index}", "", page_text, ""])
        else:
            sections.extend([page_text, ""])

    if extracted_page_count == 0:
        raise ValueError(f"No extractable text was found in {pdf_path.name}")

    output_path.write_text("\n".join(sections).strip() + "\n", encoding="utf-8")
    return output_path


def main() -> int:
    args = parse_args()
    output_path = convert_pdf_to_markdown(
        args.pdf_path,
        output_path=args.output,
        include_page_headings=not args.no_page_headings,
    )
    print(f"Markdown written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
