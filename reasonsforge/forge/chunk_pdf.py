"""Chunk a PDF paper into section-based source documents."""

import re
import sys
from pathlib import Path


def extract_text_by_page(pdf_path: Path) -> list[str]:
    """Extract text from each page of a PDF using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError:
        print("ERROR: pypdf is required for chunk-pdf.")
        print("Install: uv pip install pypdf")
        sys.exit(1)

    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return pages


def check_text_quality(pages: list[str]) -> bool:
    """Detect scanned PDFs with no text layer."""
    total_chars = sum(len(p.strip()) for p in pages)
    avg_chars = total_chars / len(pages) if pages else 0
    return avg_chars > 100


SECTION_PATTERNS = [
    # §N. Title (AGM style, rendered as $N. by pypdf)
    # Stop at first period that follows a word (end of title, start of sentence)
    re.compile(r"^[§\$](\d+)\.\s+([^.]+)"),
    # N. Title or N Title (with or without period)
    re.compile(r"^(\d+)\.?\s+([A-Z][A-Za-z\s,;:\-]+)$"),
]

STANDALONE_SECTIONS = re.compile(
    r"^(ABSTRACT|INTRODUCTION|ACKNOWLEDGMENT[S]?|REFERENCES|BIBLIOGRAPHY|APPENDIX|CONCLUSION[S]?)$",
    re.IGNORECASE,
)

# Headers/footers to ignore (journal name + page number patterns)
HEADER_FOOTER = re.compile(
    r"^\d+\s*\.?\s+[A-Z]\.\s"  # "206 J. DE KLEER", "200 . j. DE KLEER"
    r"|^[A-Z\s]+\d+$"  # "PROBLEM SOLVING WITH THE ATMS 205"
    r"|^\d+\s+[A-Z\s]+$"  # "511 THE LOGIC OF THEORY CHANGE"
    r"|^[A-Z\s]+\.\s*\d+$"  # variations with dots
)


def identify_sections(pages: list[str]) -> list[dict]:
    """Find section boundaries using structural patterns in the text."""
    sections = []

    for page_idx, page_text in enumerate(pages):
        for line in page_text.split("\n"):
            line = line.strip()
            if not line or len(line) > 100:
                continue

            # Skip headers/footers
            if HEADER_FOOTER.match(line):
                continue

            # Check standalone section names (ABSTRACT, REFERENCES, etc.)
            m = STANDALONE_SECTIONS.match(line)
            if m:
                title = line.title()
                # Use "0" for abstract, "R" for references, etc.
                number = "0" if "abstract" in title.lower() else title[0]
                sections.append({
                    "number": number,
                    "title": title,
                    "start_page": page_idx + 1,
                    "line": line,
                })
                continue

            # Check numbered section patterns
            for pattern in SECTION_PATTERNS:
                m = pattern.match(line)
                if m:
                    number = m.group(1)
                    title = m.group(2).strip().rstrip(".")
                    sections.append({
                        "number": number,
                        "title": title,
                        "start_page": page_idx + 1,
                        "line": line,
                    })
                    break

    # Filter out false positives: numbered items with implausibly high numbers
    # (real top-level sections rarely exceed 20)
    sections = [
        s for s in sections
        if not s["number"].isdigit() or int(s["number"]) <= 20
    ]

    # Compute end pages: each section ends where the next one starts
    for i in range(len(sections) - 1):
        sections[i]["end_page"] = sections[i + 1]["start_page"]
    if sections:
        sections[-1]["end_page"] = len(pages)

    return sections


def slugify(text: str) -> str:
    """Convert text to kebab-case slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")[:60]


def format_section_content(
    pages: list[str],
    section: dict,
    source_label: str,
) -> str:
    """Format raw page text into an entry."""
    start = section["start_page"] - 1
    end = section["end_page"]
    raw_pages = pages[start:end]

    header = (
        f"**Source:** {source_label}, pp. {section['start_page']}-{section['end_page']}\n\n"
    )

    body = ""
    for i, page_text in enumerate(raw_pages):
        page_num = start + i + 1
        body += f"[Page {page_num}]\n\n{page_text}\n\n"

    return header + body.rstrip() + "\n"


def make_entry_filename(prefix: str, section: dict) -> str:
    """Generate entry filename: {prefix}-s{number}-{slug}."""
    title_slug = slugify(section["title"])
    return f"{prefix}-s{section['number']}-{title_slug}"


def cmd_chunk_pdf(args):
    """Chunk a PDF paper into section-based source documents."""
    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        sys.exit(1)

    if pdf_path.suffix.lower() != ".pdf":
        print(f"Not a PDF file: {pdf_path}")
        sys.exit(1)

    prefix = args.prefix or slugify(pdf_path.stem)
    source_label = args.source_label or pdf_path.stem

    print(f"Reading PDF: {pdf_path}")
    pages = extract_text_by_page(pdf_path)
    print(f"  {len(pages)} pages extracted")

    if not pages:
        print("ERROR: No pages found in PDF.")
        sys.exit(1)

    if not check_text_quality(pages):
        print("ERROR: PDF appears to be scanned with no text layer.")
        print("OCR the PDF first (e.g., with ocrmypdf) and try again.")
        sys.exit(1)

    print("Identifying sections...")
    sections = identify_sections(pages)

    if not sections:
        print("  No sections found. Falling back to one entry per page.")
        sections = [
            {"number": str(i + 1), "title": f"Page {i + 1}", "start_page": i + 1, "end_page": i + 1}
            for i in range(len(pages))
        ]

    print(f"  Found {len(sections)} sections:")
    for s in sections:
        print(f"    {s['number']}. {s['title']} (pp. {s['start_page']}-{s['end_page']})")

    if args.dry_run:
        print("\n(dry run — no entries created)")
        return

    manifest = Path(f".chunked-{prefix}")
    done = set()
    if manifest.exists():
        done = set(manifest.read_text().strip().split("\n"))

    created = 0
    skipped = 0

    for section in sections:
        filename = make_entry_filename(prefix, section)

        if filename in done:
            print(f"  SKIP (already chunked): {filename}")
            skipped += 1
            continue

        print(f"  Creating: Section {section['number']} — {section['title']}...")

        content = format_section_content(pages, section, source_label)
        title = f"Section {section['number']}: {section['title']}"

        chunk_dir = Path("sources") / "chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        fm_lines = [
            f"source: {pdf_path}",
            f"source_label: {source_label}",
            f"section: {section['number']}",
        ]
        frontmatter = "---\n" + "\n".join(fm_lines) + "\n---\n\n"

        chunk_path = chunk_dir / f"{filename}.md"
        chunk_path.write_text(f"# {title}\n\n{frontmatter}{content}\n")
        print(f"    -> {chunk_path}")

        with manifest.open("a") as f:
            f.write(f"{filename}\n")
        done.add(filename)

        created += 1

    print(f"\nChunked {created} sections ({skipped} already done)")
    if created:
        print("Next: reasonsforge forge propose-beliefs")
