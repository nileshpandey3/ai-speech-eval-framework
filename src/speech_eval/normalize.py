# normalization policy version
NORMALIZATION_VERSION = 'v1'

def normalize_text(text: str) -> str:
    """Ignore case, commas, periods, and extra whitespace."""
    text = text.lower()
    text = text.replace(",", "").replace(".", "")

    return " ".join(text.split())
