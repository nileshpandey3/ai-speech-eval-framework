

def normalize_text(text: str) -> str:
    """Ignore case, commas, periods, and extra whitespace."""
    text = text.lower()
    text = text.replace(",", "").replace(".", "")

    return " ".join(text.split())
