def chunk_text(text, chunk_size=1000, overlap=200):
    """
    Split text into overlapping chunks.

    Args:
        text: Text to split.
        chunk_size: Maximum characters per chunk.
        overlap: Number of overlapping characters.

    Returns:
        List of text chunks.
    """

    text = text.strip()

    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks