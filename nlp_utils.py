import re


_PHRASE_NORMALIZATION = {
    "b tech": "btech",
    "b.tech": "btech",
    "b tech cse": "btech cse",
    "ai ml": "aiml",
    "ai/ml": "aiml",
    "cse ai ml": "cse aiml",
    "what's": "what is",
    "whats": "what is",
    "can i": "can i",
    # Typo normalization for common misspellings
    "compairson": "comparison",
    "comparision": "comparison",
    "comparason": "comparison",
    "comparision": "comparison",
    "elgibility": "eligibility",
    "elibility": "eligibility",
    "ellgibility": "eligibility",
    "admision": "admission",
    "addmission": "admission",
    "scolarship": "scholarship",
    "scholorship": "scholarship",
    "placment": "placement",
    "plaacement": "placement",
    "curricullum": "curriculum",
    "curriculam": "curriculum",
    "sylabbus": "syllabus",
    "sllabus": "syllabus",
    "facalty": "faculty",
    "facuity": "faculty",
    "hostle": "hostel",
    "transpotation": "transportation",
    "trnasportation": "transportation",
}


def _repair_split_words(text):
    """Fix common mid-word spaces like 'Libera lArts', 'fo rDesign'.
    Call this on lowercased but pre-cleaned text."""
    repairs = [
        (r"\bfo\s+r\b", "for"),
        (r"\bfo\s+rdesign\b", "for design"),   # "fo rDesign" -> lowercases to "fo rdesign"
        (r"\bfo\s+rhospitality\b", "for hospitality"),
        (r"\bfo\s+rliberal\b", "for liberal"),
        (r"\blibera\s+l\s+arts\b", "liberal arts"),
        (r"\blibera\s+larts\b", "liberal arts"),
        (r"\bhospita\s+lity\b", "hospitality"),
        (r"\badmis\s+sion\b", "admission"),
        (r"\beligibil\s+ity\b", "eligibility"),
        (r"\bscholar\s+ship\b", "scholarship"),
        (r"\bplace\s+ment\b", "placement"),
        (r"\bcurricul\s+um\b", "curriculum"),
        (r"\bdesig\s+n\b", "design"),
    ]
    for pattern, replacement in repairs:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def preprocess_text(text):
    """Normalize user text for vectorization and model inference."""
    normalized = text.lower().strip()
    # Apply split repairs BEFORE stripping special chars (word boundaries still intact)
    normalized = _repair_split_words(normalized)
    for source, target in _PHRASE_NORMALIZATION.items():
        normalized = normalized.replace(source, target)
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()