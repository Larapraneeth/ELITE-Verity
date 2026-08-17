from pathlib import Path
import re
import csv

from app.services.pdf_parser import extract_pdf_content
from app.services.chunker import chunk_text


def is_heading(line):
    line = line.strip()

    if not line:
        return False

    if len(line) > 120:
        return False

    if re.match(
        r"^(?:[IVXLCDM]+\.|[0-9]+(?:\.[0-9]+)*\.?)\s+[A-Z]",
        line
    ):
        return True

    if line.isupper() and 3 <= len(line.split()) <= 12:
        return True

    if re.match(
        r"^(Introduction|Methodology|Methods|Results|Conclusion|"
        r"Discussion|References|Abstract|Overview|Installation|"
        r"Operation|Troubleshooting|Specifications|Safety|"
        r"Safety Instructions|Getting Started|User Guide|User Manual|"
        r"Features|Contents)$",
        line,
        re.IGNORECASE
    ):
        return True

    return False


def detect_sections(text):
    lines = text.splitlines()

    sections = []
    current_section = "General"

    for line in lines:

        cleaned = line.strip()

        if is_heading(cleaned):
            current_section = cleaned

        sections.append(current_section)

    return lines, sections


def process_pdf(pdf_path):
    pdf_path = Path(pdf_path)

    pages = extract_pdf_content(
        pdf_path
    )

    chunks = []

    for page_data in pages:

        page_number = page_data["page"]
        text = page_data["text"]

        if not text.strip():
            continue

        lines, line_sections = detect_sections(
            text
        )

        section_blocks = []

        current_section = "General"
        current_text = []

        for line, section in zip(
            lines,
            line_sections
        ):

            if (
                section != current_section
                and current_text
            ):

                section_blocks.append(
                    {
                        "section": current_section,
                        "text": "\n".join(
                            current_text
                        )
                    }
                )

                current_text = []

            current_section = section

            if line.strip():
                current_text.append(line)

        if current_text:

            section_blocks.append(
                {
                    "section": current_section,
                    "text": "\n".join(
                        current_text
                    )
                }
            )

        chunk_number = 1

        for block in section_blocks:

            section = block["section"]
            section_text = block["text"]

            page_chunks = chunk_text(
                section_text
            )

            for chunk in page_chunks:

                chunks.append(
                    {
                        "text": chunk,
                        "filename": pdf_path.name,
                        "page": page_number,
                        "section": section,
                        "file_type": "pdf",
                        "chunk_id": (
                            f"{pdf_path.stem}"
                            f"_page{page_number}"
                            f"_chunk{chunk_number}"
                        )
                    }
                )

                chunk_number += 1

    return chunks


def process_txt(file_path):
    file_path = Path(file_path)

    text = file_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    if not text.strip():
        return []

    lines, line_sections = detect_sections(
        text
    )

    section_blocks = []

    current_section = "General"
    current_text = []

    for line, section in zip(
        lines,
        line_sections
    ):

        if (
            section != current_section
            and current_text
        ):

            section_blocks.append(
                {
                    "section": current_section,
                    "text": "\n".join(
                        current_text
                    )
                }
            )

            current_text = []

        current_section = section

        if line.strip():
            current_text.append(line)

    if current_text:

        section_blocks.append(
            {
                "section": current_section,
                "text": "\n".join(
                    current_text
                )
            }
        )

    chunks = []
    chunk_number = 1

    for block in section_blocks:

        page_chunks = chunk_text(
            block["text"]
        )

        for chunk in page_chunks:

            chunks.append(
                {
                    "text": chunk,
                    "filename": file_path.name,
                    "page": 1,
                    "section": block["section"],
                    "file_type": "txt",
                    "chunk_id": (
                        f"{file_path.stem}"
                        f"_chunk{chunk_number}"
                    )
                }
            )

            chunk_number += 1

    return chunks


def process_csv(file_path):
    file_path = Path(file_path)

    rows = []

    with open(
        file_path,
        "r",
        encoding="utf-8-sig",
        errors="ignore",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        if not reader.fieldnames:
            return []

        for row_number, row in enumerate(
            reader,
            start=2
        ):

            values = []

            for column, value in row.items():

                if value is None:
                    continue

                value = str(value).strip()

                if not value:
                    continue

                values.append(
                    f"{column}: {value}"
                )

            if values:

                rows.append(
                    {
                        "row_number": row_number,
                        "text": " | ".join(values)
                    }
                )

    chunks = []
    chunk_number = 1

    for row in rows:

        row_chunks = chunk_text(
            row["text"]
        )

        for chunk in row_chunks:

            chunks.append(
                {
                    "text": chunk,
                    "filename": file_path.name,
                    "page": row["row_number"],
                    "section": "CSV Data",
                    "file_type": "csv",
                    "sheet": "",
                    "chunk_id": (
                        f"{file_path.stem}"
                        f"_row{row['row_number']}"
                        f"_chunk{chunk_number}"
                    )
                }
            )

            chunk_number += 1

    return chunks


def process_excel(file_path):
    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError(
            "pandas is required for Excel processing."
        )

    file_path = Path(file_path)

    excel_file = pd.ExcelFile(
        file_path
    )

    chunks = []
    chunk_number = 1

    for sheet_name in excel_file.sheet_names:

        dataframe = pd.read_excel(
            file_path,
            sheet_name=sheet_name
        )

        dataframe = dataframe.fillna("")

        for row_number, row in dataframe.iterrows():

            values = []

            for column in dataframe.columns:

                value = str(
                    row[column]
                ).strip()

                if not value:
                    continue

                values.append(
                    f"{column}: {value}"
                )

            if not values:
                continue

            text = " | ".join(values)

            row_chunks = chunk_text(
                text
            )

            for chunk in row_chunks:

                chunks.append(
                    {
                        "text": chunk,
                        "filename": file_path.name,
                        "page": row_number + 2,
                        "section": "Excel Data",
                        "file_type": "excel",
                        "sheet": str(
                            sheet_name
                        ),
                        "chunk_id": (
                            f"{file_path.stem}"
                            f"_{sheet_name}"
                            f"_row{row_number + 2}"
                            f"_chunk{chunk_number}"
                        )
                    }
                )

                chunk_number += 1

    return chunks


def process_document(file_path):
    file_path = Path(file_path)

    extension = file_path.suffix.lower()

    if extension == ".pdf":

        return process_pdf(
            file_path
        )

    if extension == ".txt":

        return process_txt(
            file_path
        )

    if extension == ".csv":

        return process_csv(
            file_path
        )

    if extension in [
        ".xlsx",
        ".xlsm",
        ".xls"
    ]:

        return process_excel(
            file_path
        )

    raise ValueError(
        f"Unsupported file type: {extension}"
    )