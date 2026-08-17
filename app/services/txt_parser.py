from pathlib import Path


def extract_txt_content(txt_path):
    txt_path = Path(txt_path)

    text = txt_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    return [
        {
            "page": 1,
            "text": text
        }
    ]