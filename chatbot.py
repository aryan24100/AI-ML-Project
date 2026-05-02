import json
import os
import pickle
import random
import re
from scipy.sparse import hstack
from sklearn.metrics.pairwise import cosine_similarity

from nlp_utils import preprocess_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
INTENTS_PATH = os.path.join(BASE_DIR, "intents.json")

CONFIDENCE_THRESHOLD = 0.42
HIGH_SIMILARITY_THRESHOLD = 0.56
LOW_CONFIDENCE_SIMILARITY_THRESHOLD = 0.38
LOW_CONFIDENCE_CLASSIFIER_MIN = 0.20
BRAIN_CLASSIFIER_WEIGHT = 0.56
BRAIN_PATTERN_WEIGHT = 0.34
BRAIN_KEYWORD_WEIGHT = 0.18
BRAIN_SCORE_THRESHOLD = 0.34
BRAIN_MARGIN_THRESHOLD = 0.04
YES_WORDS = {
    "yes",
    "yess",
    "yeah",
    "yup",
    "ok",
    "okay",
    "haan",
    "yes please",
}

GREETING_WORDS = {
    "hi",
    "hii",
    "hiii",
    "hello",
    "hey",
    "heyy",
}

GOODBYE_WORDS = {
    "bye",
    "byee",
    "goodbye",
    "see you",
    "see ya",
    "tata",
}
SHORT_QUERY_ALLOWLIST = {
    "fees",
    "fee",
    "hostel",
    "placement",
    "placements",
    "admission",
    "courses",
    "course",
    "scholarship",
    "scholarships",
    "contact",
    "hello",
    "hi",
    "hey",
    "bye",
    "goodbye",
    "mba",
    "bba",
    "btech",
    "law",
    "design",
    "hospitality",
    "internship",
    "internships",
    "curriculum",
    "syllabus",
    "campus",
    "hostel",
    "mess",
    "canteen",
    "transport",
    "faculty",
    "placements",
    "eligibility",
}

ELIGIBILITY_HINT_WORDS = {
    "eligibility",
    "eligible",
    "criteria",
    "requirement",
    "requirements",
    "qualify",
    "qualification",
}

ELIGIBILITY_COURSE_TAGS = {
    "eligibility_criteria_btech": {"btech", "engineering", "cse", "aiml"},
    "eligibility_criteria_mba": {"mba", "cat", "mat"},
    "eligibility_criteria_bba": {"bba"},
    "eligibility_criteria_law": {"law", "llb", "clat"},
}

FOLLOWUP_BTECH_COMPARISON = "btech_comparison_details"

COURSE_FEES_FOLLOWUP = "course_fees"

COURSE_TAG_TO_KEY = {
    "btech_details": "btech",
    "eligibility_criteria_btech": "btech",
    "eligibility_criteria_mba": "mba",
    "eligibility_criteria_bba": "bba",
    "eligibility_criteria_law": "law",
    "liberal_arts_info": "liberal_arts",
    "design_fashion_info": "design",
    "hospitality_info": "hospitality",
    "courses_offered": None,
}

COURSE_FOLLOWUP_PROMPTS = {
    "btech": "If you want, I can share the BTech fee breakdown next.",
    "mba": "If you want, I can share the MBA fee details next.",
    "bba": "If you want, I can share the BBA fee details next.",
    "law": "If you want, I can share the Law fee details next.",
    "liberal_arts": "If you want, I can share the Liberal Arts fee details next.",
    "design": "If you want, I can share the Design & Fashion fee details next.",
    "hospitality": "If you want, I can share the Hospitality fee details next.",
}

COURSE_FEE_OVERVIEWS = {
    "btech": "BTech fees depend on the admission cycle and program type. I can also break it into tuition, hostel, and scholarship parts if you want.",
    "mba": "MBA fees depend on the admission cycle and program structure. I can also break it into tuition and scholarship parts if you want.",
    "bba": "BBA fees depend on the admission cycle and program structure. I can also break it into tuition and scholarship parts if you want.",
    "law": "Law fees depend on the admission cycle and the exact program. I can also break it into tuition and scholarship parts if you want.",
    "liberal_arts": "Liberal Arts fees depend on the admission cycle and program path. I can also break it into tuition and scholarship parts if you want.",
    "design": "Design & Fashion fees depend on the admission cycle and the exact program. I can also break it into tuition and scholarship parts if you want.",
    "hospitality": "Hospitality fees depend on the admission cycle and the exact program. I can also break it into tuition and scholarship parts if you want.",
}

COMBINE_CONNECTOR_PATTERNS = [
    r"\band\b",
    r"\balso\b",
    r"\bplus\b",
    r"\bas well as\b",
    r"\bwith\b",
    r"&",
    r",",
]

NON_COMBINABLE_TAGS = {"greeting", "goodbye", "fallback", "faq_general"}

TAG_TITLES = {
    "admission_process": "Admission Process",
    "eligibility_criteria_general": "General Eligibility",
    "eligibility_criteria_btech": "BTech Eligibility",
    "eligibility_criteria_mba": "MBA Eligibility",
    "eligibility_criteria_bba": "BBA Eligibility",
    "eligibility_criteria_law": "Law Eligibility",
    "courses_offered": "Courses Offered",
    "btech_details": "BTech Details",
    "btech_camparison": "BTech Comparison",
    "placements_overview": "Placements",
    "contact_info": "Contact",
    "scholarship_info": "Scholarships",
    "campus_location": "Campus Location",
    "courses_curriculum": "Curriculum",
    "hostel_info": "Hostel",
    "mess_info": "Mess",
    "transportation_info": "Transportation",
    "counselling_support": "Counselling Support",
    "sports_facilities": "Sports",
    "events_and_clubs": "Events and Clubs",
    "international_programs": "International Programs",
    "faculty_info": "Faculty",
    "canteen_food": "Canteen",
    "fees_general": "Fees",
    "internship_info": "Internships",
    "college_timings": "College Timings",
    "website_navigation": "Website Navigation",
    "facilities": "Facilities",
    "liberal_arts_info": "Liberal Arts",
    "design_fashion_info": "Design and Fashion",
    "hospitality_info": "Hospitality",
}

QUESTION_CUE_WORDS = {
    "what",
    "how",
    "when",
    "where",
    "who",
    "which",
    "does",
    "do",
    "can",
    "is",
    "are",
}

INTENT_KEYWORDS = {
    "internship_info": {"internship": 1.0, "internships": 1.0, "intern": 0.8},
    "placements_overview": {"placement": 1.0, "placements": 1.0, "package": 0.8, "job": 0.8},
    "transportation_info": {"transport": 1.0, "bus": 0.9, "metro": 0.9, "shuttle": 0.9, "route": 0.8},
    "campus_location": {"location": 1.0, "address": 1.0, "where": 0.6, "campus": 0.6},
    "fees_general": {"fee": 1.0, "fees": 1.0, "cost": 0.8, "expense": 0.8, "tuition": 0.8},
    "admission_process": {"admission": 1.0, "apply": 0.9, "application": 0.9, "register": 0.8},
    "contact_info": {"contact": 1.0, "phone": 0.9, "helpline": 0.9, "email": 0.9, "call": 0.8},
    "scholarship_info": {"scholarship": 1.0, "scholarships": 1.0, "waiver": 0.8},
    "hostel_info": {"hostel": 1.0, "room": 0.7, "accommodation": 0.8, "pg": 0.6},
    "courses_offered": {"course": 1.0, "courses": 1.0, "program": 0.9, "programs": 0.9},
    "courses_curriculum": {"curriculum": 1.0, "syllabus": 1.0, "subjects": 0.9, "semester": 0.8, "subject": 0.9},
    "btech_camparison": {"comparison": 1.0, "compare": 1.0, "difference": 0.9, "versus": 0.8, "vs": 0.8, "regular": 0.6},
    "faculty_info": {"faculty": 1.0, "professor": 0.9, "teacher": 0.9, "staff": 0.7},
    "sports_facilities": {"sports": 1.0, "gym": 0.9, "fitness": 0.8, "cricket": 0.8, "basketball": 0.8},
    "events_and_clubs": {"club": 1.0, "clubs": 1.0, "event": 0.9, "events": 0.9, "cultural": 0.8},
    "canteen_food": {"canteen": 1.0, "cafeteria": 0.9},
    "mess_info": {"mess": 1.0, "vegetarian": 0.8},
    "college_timings": {"timing": 1.0, "timings": 1.0, "hours": 0.8, "schedule": 0.8},
    "counselling_support": {"counselling": 1.0, "counseling": 1.0, "guidance": 0.9},
    "international_programs": {"international": 1.0, "global": 0.9, "abroad": 1.0, "exchange": 0.8},
    "liberal_arts_info": {"liberal": 1.0, "arts": 0.8},
    "design_fashion_info": {"design": 1.0, "fashion": 1.0},
    "hospitality_info": {"hospitality": 1.0},
}

INTENT_NEXT_TAGS = {
    "admission_process": ["eligibility_criteria_general", "fees_general", "scholarship_info", "contact_info"],
    "eligibility_criteria_general": ["eligibility_criteria_btech", "eligibility_criteria_mba", "eligibility_criteria_bba"],
    "eligibility_criteria_btech": ["btech_details", "fees_general", "placements_overview"],
    "btech_details": ["btech_camparison", "fees_general", "placements_overview"],
    "btech_camparison": ["fees_general", "placements_overview", "scholarship_info"],
    "courses_offered": ["eligibility_criteria_general", "fees_general", "placements_overview", "liberal_arts_info", "design_fashion_info", "hospitality_info"],
    "fees_general": ["scholarship_info", "admission_process", "contact_info"],
    "placements_overview": ["internship_info", "courses_curriculum", "btech_details"],
    "hostel_info": ["fees_general", "transportation_info", "facilities"],
    "scholarship_info": ["admission_process", "eligibility_criteria_general", "contact_info"],
    "contact_info": ["admission_process", "courses_offered", "fees_general"],
}

DEFAULT_GENERAL_SUGGESTIONS = [
    "What is the admission process?",
    "What courses are offered?",
    "What is the fee structure?",
    "How are placements and internships at IILM?",
    "How can I contact the admission cell?",
]

BTECH_COMPARISON_DETAILED_REPLY = (
    "Great choice. Here is a detailed comparison:\n\n"
    "1) Regular BTech CSE\n"
    "- Focus: Core CS fundamentals (DSA, DBMS, OS, Networks, Programming)\n"
    "- Best for: Students who want strong base and flexibility for future domains\n"
    "- Cost: Lower than collaboration tracks\n"
    "- Placement impact: Good, but depends more on self-driven projects and skills\n\n"
    "2) Collaboration BTech (IBM / Apple / Microsoft tracks)\n"
    "- Focus: Industry-aligned curriculum with practical tools and certifications\n"
    "- Best for: Students who want early industry exposure and job-ready skill depth\n"
    "- Cost: Higher, due to partner-led training and specialization modules\n"
    "- Placement impact: Often stronger profile for role-specific hiring\n\n"
    "3) Subject-wise difference (typical)\n"
    "- Regular: More theory + standard labs\n"
    "- Collaboration: More applied labs, partner content, real-world mini projects\n\n"
    "4) Which one should you choose?\n"
    "- Choose Regular BTech if budget and fundamentals are your top priority.\n"
    "- Choose Collaboration BTech if you want stronger industry alignment and are comfortable with higher fees.\n\n"
    "If you want, I can next give a semester-wise roadmap for both options."
)

BTECH_DETAIL_FOLLOWUPS = [
    "What is BTech eligibility criteria in detail?",
    "What is the BTech fee structure and scholarship options?",
    "What is the difference between regular and collaboration BTech?",
    "What specializations are available and how to choose one?",
    "How are BTech placements and internship opportunities?",
]

BTECH_SPECIALIZATION_REPLY = (
    "For BTech specializations, students usually explore options like CSE, AI/ML, Cloud Computing, Cybersecurity, and related software-focused tracks. "
    "The exact availability can vary by admission cycle, so it is best to verify the current list on the official programs page: https://iilm.edu/programmes/"
)

# Load files
with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)
with open(VECTORIZER_PATH, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

with open(INTENTS_PATH, encoding="utf-8") as file:
    data = json.load(file)


def _intent_example_map():
    examples = {}
    for intent in data["intents"]:
        patterns = intent.get("patterns", [])
        if patterns:
            examples[intent["tag"]] = patterns[0]
    return examples


INTENT_EXAMPLES = _intent_example_map()


def _vectorize_texts(texts):
    # Backward-compatible loading: vectorizer may be a single vectorizer or a dict bundle.
    if isinstance(vectorizer, dict) and "word" in vectorizer and "char" in vectorizer:
        word_features = vectorizer["word"].transform(texts)
        char_features = vectorizer["char"].transform(texts)
        return hstack([word_features, char_features]).tocsr()
    return vectorizer.transform(texts)


def _build_pattern_index():
    pattern_texts = []
    pattern_tags = []
    for intent in data["intents"]:
        tag = intent.get("tag")
        for pattern in intent.get("patterns", []):
            cleaned_pattern = preprocess_text(pattern)
            if cleaned_pattern:
                pattern_texts.append(cleaned_pattern)
                pattern_tags.append(tag)

    if not pattern_texts:
        return [], []

    pattern_vectors = _vectorize_texts(pattern_texts)
    return pattern_vectors, pattern_tags


PATTERN_VECTORS, PATTERN_TAGS = _build_pattern_index()


def _predict_intent(cleaned_text):
    features = _vectorize_texts([cleaned_text])
    probabilities = model.predict_proba(features)[0]
    best_index = int(probabilities.argmax())
    predicted_tag = model.classes_[best_index]
    confidence = float(probabilities[best_index])
    return predicted_tag, confidence


def _classifier_scores(cleaned_text):
    features = _vectorize_texts([cleaned_text])
    probabilities = model.predict_proba(features)[0]
    return {model.classes_[i]: float(probabilities[i]) for i in range(len(model.classes_))}


def _find_response_by_tag(tag):
    for intent in data["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent.get("responses", []))
    return None


def _all_responses_for_tag(tag):
    for intent in data["intents"]:
        if intent.get("tag") == tag:
            return intent.get("responses", [])
    return []


def _extract_admission_contact():
    contact_intent = next((intent for intent in data["intents"] if intent.get("tag") == "contact_info"), None)
    if not contact_intent:
        return "+91-080 6590 5220", "admissions@iilm.edu"

    phone_pattern = re.compile(r"\+?\d[\d\s\-]{8,}\d")
    email_pattern = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

    found_phone = None
    found_email = None
    for response in contact_intent.get("responses", []):
        if not found_phone:
            phone_match = phone_pattern.search(response)
            if phone_match:
                found_phone = phone_match.group(0).strip()
        if not found_email:
            email_match = email_pattern.search(response)
            if email_match:
                found_email = email_match.group(0).strip()
        if found_phone and found_email:
            break

    return found_phone or "+91-080 6590 5220", found_email or "admissions@iilm.edu"


_CLEAN_SAMPLE_QUESTIONS = [
    "What is the admission process?",
    "What courses does IILM offer?",
    "What is the fee structure?",
    "Am I eligible for BTech?",
    "How can I contact admissions?",
]


def _uncertain_with_contact_message(sample_questions=None):
    phone, email = _extract_admission_contact()
    questions_to_show = _CLEAN_SAMPLE_QUESTIONS[:3]
    base_message = "I did not fully understand that. You can try asking like:"
    question_list = " | ".join(questions_to_show)
    return (
        f"{base_message} {question_list}\n"
        f"For direct help, contact the admission cell at {phone} or {email}."
    )


def _tag_title(tag):
    return TAG_TITLES.get(tag, tag.replace("_", " ").title())


def _select_best_response_for_query(tag, user_input):
    responses = _all_responses_for_tag(tag)
    if not responses:
        return None
    if len(responses) == 1:
        return responses[0]

    cleaned_query = preprocess_text(user_input)
    if not cleaned_query:
        return responses[0]  # Default to first (overview) response

    # For general "tell me about / what is / overview" queries, prefer the first response
    OVERVIEW_SIGNALS = {"tell me about", "what is", "about", "overview", "details", "details about", "tell me more", "explain"}
    is_overview_query = any(sig in cleaned_query for sig in OVERVIEW_SIGNALS)
    specific_signals  = {"fee", "fees", "cost", "eligib", "apply", "contact", "scholarship", "hostel", "placement"}
    has_specific_term = any(term in cleaned_query for term in specific_signals)

    if is_overview_query and not has_specific_term:
        return responses[0]  # Return the overview/first response

    response_vectors = _vectorize_texts([preprocess_text(r) for r in responses])
    query_vector = _vectorize_texts([cleaned_query])
    similarities = cosine_similarity(query_vector, response_vectors)[0]

    query_tokens = set(cleaned_query.split())
    scored = []
    for idx, response in enumerate(responses):
        cleaned_response = preprocess_text(response)
        response_tokens  = set(cleaned_response.split())
        token_overlap    = len(query_tokens & response_tokens)
        overlap_score    = 0.0 if not query_tokens else token_overlap / max(1, len(query_tokens))
        semantic_score   = float(similarities[idx])
        combined = (0.75 * semantic_score) + (0.25 * overlap_score)
        scored.append((combined, idx, response))

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][2]


def _humanize_response(tag, user_input, base_response):
    if not base_response:
        return base_response

    cleaned_query = preprocess_text(user_input)
    starts_with_yes_no_form = cleaned_query.startswith(("is ", "are ", "can ", "does ", "do ", "will "))
    response_starts_with_affirmation = preprocess_text(base_response).startswith(("yes", "no"))

    direct_openers = {
        "internship_info": "Yes, IILM supports students with internship opportunities.",
        "placements_overview": "IILM provides placement support for students across major programs.",
        "hostel_info": "Yes, hostel facilities are available for students.",
        "transportation_info": "Yes, transportation support is available for students.",
        "contact_info": "You can directly connect with the admission team for official guidance.",
        "fees_general": "Fees depend on your program and campus choices.",
        "admission_process": "Admissions are handled through a step-by-step application process.",
        "liberal_arts_info": "Yes, Liberal Arts is offered at IILM.",
        "design_fashion_info": "Yes, Design & Fashion is offered at IILM.",
        "hospitality_info": "Yes, Hospitality is offered at IILM.",
        "events_and_clubs": "Yes, IILM has student clubs and events.",
        "sports_facilities": "Yes, sports facilities are available on campus.",
        "counselling_support": "Yes, admission counselling support is available.",
        "international_programs": "Yes, IILM offers international exposure programs.",
    }

    opener = direct_openers.get(tag)
    if (
        starts_with_yes_no_form
        and opener
        and opener.lower() not in base_response.lower()
        and not response_starts_with_affirmation
    ):
        return f"{opener} {base_response}"

    if tag == "contact_info":
        phone, email = _extract_admission_contact()
        phone_present = phone and phone in base_response
        email_present = email and email in base_response
        if not phone_present and not email_present:
            return f"{base_response}\nContact: {phone} | {email}"
        return base_response

    # Keep responses conversational but concise for common informational asks.
    if any(phrase in cleaned_query for phrase in {"tell me", "explain", "i want to know", "details"}):
        if not base_response.startswith("Sure"):
            return f"Sure, here is what I can share: {base_response}"

    return base_response


def _nearest_pattern_tag(cleaned_text):
    if not PATTERN_TAGS:
        return None, 0.0

    query_vector = _vectorize_texts([cleaned_text])
    similarities = cosine_similarity(query_vector, PATTERN_VECTORS)[0]
    best_index = int(similarities.argmax())
    return PATTERN_TAGS[best_index], float(similarities[best_index])


def _pattern_scores_by_tag(cleaned_text):
    if not PATTERN_TAGS:
        return {}

    query_vector = _vectorize_texts([cleaned_text])
    similarities = cosine_similarity(query_vector, PATTERN_VECTORS)[0]
    by_tag = {}

    for idx, tag in enumerate(PATTERN_TAGS):
        score = float(similarities[idx])
        previous = by_tag.get(tag, 0.0)
        if score > previous:
            by_tag[tag] = score

    return by_tag


def _keyword_scores_by_tag(cleaned_text):
    scores = {}
    for tag, weights in INTENT_KEYWORDS.items():
        matched_weight = 0.0
        total_weight = 0.0
        for keyword, weight in weights.items():
            total_weight += weight
            if keyword in cleaned_text:
                matched_weight += weight

        if total_weight > 0.0 and matched_weight > 0.0:
            scores[tag] = min(1.0, matched_weight / total_weight)

    return scores


def _focus_user_query(user_input):
    text = user_input.strip()
    if not text:
        return text

    # Prefer the most question-like clause when users paste extra lines or copied text.
    chunks = [chunk.strip() for chunk in re.split(r"[\n\r]+", text) if chunk.strip()]
    if len(chunks) == 1:
        chunks = [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+", text) if chunk.strip()]

    if len(chunks) <= 1:
        return text

    best_chunk = chunks[0]
    best_score = -1.0
    for idx, chunk in enumerate(chunks):
        cleaned_chunk = preprocess_text(chunk)
        if not cleaned_chunk:
            continue

        score = 0.0
        tokens = cleaned_chunk.split()
        if "?" in chunk:
            score += 1.8
        if tokens and tokens[0] in QUESTION_CUE_WORDS:
            score += 1.2
        if any(cue in cleaned_chunk for cue in {"i want", "tell me", "can you", "help me"}):
            score += 0.8

        # Earlier chunks are more likely to be the user's actual query.
        score += max(0.0, 0.35 - (idx * 0.08))

        if score > best_score:
            best_score = score
            best_chunk = chunk

    return best_chunk


def _brain_select_intent(cleaned_text):
    classifier_scores = _classifier_scores(cleaned_text)
    pattern_scores = _pattern_scores_by_tag(cleaned_text)
    keyword_scores = _keyword_scores_by_tag(cleaned_text)

    all_tags = set(classifier_scores) | set(pattern_scores) | set(keyword_scores)
    if not all_tags:
        return None

    ranked = []
    for tag in all_tags:
        classifier_score = classifier_scores.get(tag, 0.0)
        pattern_score = pattern_scores.get(tag, 0.0)
        keyword_score = keyword_scores.get(tag, 0.0)

        combined_score = (
            BRAIN_CLASSIFIER_WEIGHT * classifier_score
            + BRAIN_PATTERN_WEIGHT * pattern_score
            + BRAIN_KEYWORD_WEIGHT * keyword_score
        )

        ranked.append(
            {
                "tag": tag,
                "score": combined_score,
                "classifier": classifier_score,
                "pattern": pattern_score,
                "keyword": keyword_score,
            }
        )

    ranked.sort(key=lambda item: item["score"], reverse=True)
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    margin = best["score"] - (second["score"] if second else 0.0)

    if best["score"] < BRAIN_SCORE_THRESHOLD and best["classifier"] < CONFIDENCE_THRESHOLD:
        return {
            "status": "uncertain",
            "tag": None,
            "confidence": best["classifier"],
            "similarity": best["pattern"],
            "method": "brain-low-score",
        }

    if margin < BRAIN_MARGIN_THRESHOLD and best["classifier"] < CONFIDENCE_THRESHOLD:
        return {
            "status": "uncertain",
            "tag": None,
            "confidence": best["classifier"],
            "similarity": best["pattern"],
            "method": "brain-low-margin",
        }

    return {
        "status": "ok",
        "tag": best["tag"],
        "confidence": best["classifier"],
        "similarity": best["pattern"],
        "method": "brain-hybrid",
    }


COURSE_SPECIFIC_INTENTS = {
    "hospitality": "hospitality_info",
    "design": "design_fashion_info",
    "fashion": "design_fashion_info",
    "liberal arts": "liberal_arts_info",
    "liberal": "liberal_arts_info",
}

# These course keywords trigger their own specific intents only when the query
# is clearly about the course itself (not eligibility/fees/admission subqueries).
COURSE_INFO_KEYWORDS = {
    "mba": "eligibility_criteria_mba",   # "What is MBA?" -> MBA eligibility as starting point
    "bba": "eligibility_criteria_bba",
}


def _rule_based_course_intent(cleaned_text):
    """Route 'Can I apply for Hospitality/Design/Liberal Arts?' to specific course intents."""
    for keyword, tag in COURSE_SPECIFIC_INTENTS.items():
        if keyword in cleaned_text:
            return tag
    return None


def _rule_based_eligibility_tag(cleaned_text):
    if "scholarship" in cleaned_text:
        return None

    has_hint_word = any(word in cleaned_text for word in ELIGIBILITY_HINT_WORDS)
    has_hint_phrase = any(
        phrase in cleaned_text
        for phrase in {
            "who can apply",
            "am i eligible",
            "basic eligibility",
            "minimum marks",
            "what marks",
        }
    )

    # "can i apply for X" alone (without eligibility words) should NOT trigger eligibility routing
    # — it's a question about admission/courses, not eligibility criteria.
    can_apply_without_eligibility = (
        "can i apply" in cleaned_text
        and not has_hint_word
        and not any(
            phrase in cleaned_text
            for phrase in {"am i eligible", "basic eligibility", "minimum marks", "what marks"}
        )
    )
    if can_apply_without_eligibility:
        return None

    if not (has_hint_word or has_hint_phrase):
        return None

    # "what courses can i apply for" should route to courses_offered, not eligibility
    if "what courses" in cleaned_text or "which courses" in cleaned_text:
        return None

    for tag, keywords in ELIGIBILITY_COURSE_TAGS.items():
        if any(keyword in cleaned_text for keyword in keywords):
            return tag

    # "foreign students eligible" or general eligibility without a specific course
    # should not fire eligibility rule if it's more of a general info query
    foreign_query = any(word in cleaned_text for word in {"foreign", "international", "nri", "overseas"})
    if foreign_query and not any(keyword in cleaned_text for kw_set in ELIGIBILITY_COURSE_TAGS.values() for keyword in kw_set):
        return None

    # Only return general eligibility if there's a clear eligibility keyword
    if has_hint_word:
        return "eligibility_criteria_general"

    return None


def predict_intent(user_input):
    if not isinstance(user_input, str):
        return {
            "status": "invalid",
            "message": "Please type your question in text format.",
            "tag": None,
            "confidence": 0.0,
            "similarity": 0.0,
        }

    focused_query = _focus_user_query(user_input)
    cleaned = preprocess_text(focused_query)

    if not cleaned:
        return {
            "status": "empty",
            "message": "Please type a question, for example: 'What is the admission process?'",
            "tag": None,
            "confidence": 0.0,
            "similarity": 0.0,
        }

    tokens = cleaned.split()
    if len(tokens) == 1 and tokens[0] not in SHORT_QUERY_ALLOWLIST:
        return {
            "status": "too_short",
            "message": (
                "Can you be a bit more specific? For example: 'BTech fees', "
                "'admission process', or 'placement details'."
            ),
            "tag": None,
            "confidence": 0.0,
            "similarity": 0.0,
        }

    rule_tag = _rule_based_eligibility_tag(cleaned)
    if rule_tag:
        # But first check if this is actually a course-info query (not eligibility)
        course_tag = _rule_based_course_intent(cleaned)
        if course_tag and not any(word in cleaned for word in ELIGIBILITY_HINT_WORDS):
            return {
                "status": "ok",
                "message": "",
                "tag": course_tag,
                "confidence": 1.0,
                "similarity": 1.0,
                "method": "rule-based-course",
            }
        return {
            "status": "ok",
            "message": "",
            "tag": rule_tag,
            "confidence": 1.0,
            "similarity": 1.0,
            "method": "rule-based-eligibility",
        }

    # Check for specific course intents (hospitality, design, liberal arts) before brain
    course_tag = _rule_based_course_intent(cleaned)
    if course_tag:
        return {
            "status": "ok",
            "message": "",
            "tag": course_tag,
            "confidence": 1.0,
            "similarity": 1.0,
            "method": "rule-based-course",
        }

    # Single-keyword decisive override: if one keyword is both high-weight and unambiguous,
    # trust it directly rather than going through brain scoring which can be noise-sensitive.
    DECISIVE_SINGLE_KEYWORDS = {
        "curriculum": "courses_curriculum",
        "syllabus": "courses_curriculum",
        "mess": "mess_info",
        "canteen": "canteen_food",
        "cafeteria": "canteen_food",
        "counselling": "counselling_support",
        "counseling": "counselling_support",
    }
    # For scholarship, only use decisive routing if it's a standalone query
    # (not combined with course names that should route to specific eligibility/fee pages)
    tokens_cleaned = cleaned.split()
    if "scholarship" in tokens_cleaned or "scholarships" in tokens_cleaned:
        DECISIVE_SINGLE_KEYWORDS["scholarship"] = "scholarship_info"
        DECISIVE_SINGLE_KEYWORDS["scholarships"] = "scholarship_info"

    for keyword, decisive_tag in DECISIVE_SINGLE_KEYWORDS.items():
        if keyword in cleaned:
            return {
                "status": "ok",
                "message": "",
                "tag": decisive_tag,
                "confidence": 1.0,
                "similarity": 1.0,
                "method": "rule-based-keyword",
            }

    # Route review/about/overview queries to faq_general
    faq_signals = {"review", "overview", "about iilm", "is iilm good", "iilm good", "tell me about iilm", "what is iilm"}
    if any(signal in cleaned for signal in faq_signals):
        return {
            "status": "ok",
            "message": "",
            "tag": "faq_general",
            "confidence": 1.0,
            "similarity": 1.0,
            "method": "rule-based-keyword",
        }

    # Route accreditation / ranking queries
    accreditation_signals = {"naac", "nirf", "ugc", "aicte", "accredit", "ranking", "rank", "affiliated", "affiliation", "deemed university", "approved", "recognition"}
    if any(sig in cleaned for sig in accreditation_signals):
        return {
            "status": "ok",
            "message": "",
            "tag": "accreditation_ranking",
            "confidence": 1.0,
            "similarity": 1.0,
            "method": "rule-based-keyword",
        }

    # Route admission date / deadline queries
    date_signals = {"last date", "deadline", "admission open", "admission close", "when to apply", "application deadline", "admission date", "last day", "admission schedule", "apply before", "open now"}
    if any(sig in cleaned for sig in date_signals):
        return {
            "status": "ok",
            "message": "",
            "tag": "admission_dates",
            "confidence": 1.0,
            "similarity": 1.0,
            "method": "rule-based-keyword",
        }

    # Route NRI / international student queries
    nri_signals = {"nri", "overseas", "foreign national", "outside india", "from abroad", "international student"}
    if any(sig in cleaned for sig in nri_signals):
        return {
            "status": "ok",
            "message": "",
            "tag": "nri_admission",
            "confidence": 1.0,
            "similarity": 1.0,
            "method": "rule-based-keyword",
        }

    # ── Faculty routing — school-specific and general ──────────────────────
    FACULTY_SCHOOL_SIGNALS = {
        "faculty_leadership":       {"dean", "director of school", "leadership", "who leads iilm", "associate dean", "vice chancellor", "pro vice", "registrar", "chancellor", "dean list"},
        "faculty_cse":              {"cse faculty", "computer science faculty", "bca faculty", "mca faculty", "teaches cse", "teaches btech cse", "teaches programming", "ai ml faculty", "school of cse", "teachers in cse", "who are the teachers in cse"},
        "faculty_engineering":      {"engineering faculty", "mechanical faculty", "electrical faculty", "electronics faculty", "biotechnology faculty", "teaches mechanical", "teaches electrical", "mtech faculty", "school of engineering faculty"},
        "faculty_law":              {"law faculty", "llb professor", "llm faculty", "teaches law", "school of law professor", "law professor", "ba llb faculty", "bba llb faculty", "law department faculty"},
        "faculty_humanities":       {"humanities faculty", "liberal arts faculty", "liberal arts professor", "psychology faculty", "communication faculty", "journalism faculty", "mass communication faculty", "teaches liberal arts", "teaches psychology", "liberal arts professors"},
        "faculty_sciences":         {"mathematics faculty", "physics faculty", "chemistry faculty", "forensic science faculty", "maths faculty", "teaches maths", "teaches mathematics", "teaches physics", "teaches chemistry", "forensic faculty", "bsc faculty", "school of sciences professor", "sciences professors"},
        "faculty_management":       {"management faculty", "mba faculty", "bba faculty", "teaches mba", "teaches business", "school of management professor", "management professor"},
        "faculty_industry_experts":  {"professor of practice", "industry expert", "visiting faculty", "guest lecturer", "industry professional", "practical faculty"},
        "faculty_quality":          {"faculty qualified", "phd faculty", "faculty ratio", "faculty to student ratio", "are teachers good", "how good is faculty", "faculty experience", "faculty research", "do faculty", "faculty have phd", "faculty ahve"},
        "faculty_specific_person":  {"who is dr.", "who is prof.", "hod of", "head of department of", "which professor teaches"},
    }
    for _ftag, _fsignals in FACULTY_SCHOOL_SIGNALS.items():
        if any(sig in cleaned for sig in _fsignals):
            return {"status": "ok", "message": "", "tag": _ftag,
                    "confidence": 1.0, "similarity": 1.0, "method": "rule-based-keyword"}

    # General faculty/teacher query — only if no specific school signal caught it
    _faculty_words = {"faculty", "professor", "professors", "lecturer", "teaching staff"}
    _faculty_exclude = {"development", "fdp", "development program", "liberal arts", "law", "cse", "engineering", "management", "sciences", "humanities", "psychology", "communication"}
    if any(w in cleaned for w in _faculty_words) and not any(e in cleaned for e in _faculty_exclude):
        return {"status": "ok", "message": "", "tag": "faculty_info",
                "confidence": 1.0, "similarity": 1.0, "method": "rule-based-keyword"}

    # "placement cell" or "placement team" should go to placements, not library
    if "placement cell" in cleaned or "placement team" in cleaned or "placement office" in cleaned:
        return {
            "status": "ok", "message": "", "tag": "placements_overview",
            "confidence": 1.0, "similarity": 1.0, "method": "rule-based-keyword",
        }

    # Route library / infrastructure queries (but NOT if it's a placement query)
    infra_signals = {"library", "parking", "reading room", "computer lab", "infrastructure"}
    wifi_signals  = {"wifi", "wi-fi", "wi fi", "password", "internet"}
    if any(sig in cleaned for sig in wifi_signals):
        return {
            "status": "ok", "message": "", "tag": "out_of_scope",
            "confidence": 1.0, "similarity": 1.0, "method": "rule-based-keyword",
        }
    if any(sig in cleaned for sig in infra_signals) and "placement" not in cleaned:
        return {
            "status": "ok", "message": "", "tag": "library_facilities",
            "confidence": 1.0, "similarity": 1.0, "method": "rule-based-keyword",
        }

    # Route course counselling / confusion queries
    counselling_signals = {"confused about", "which course", "help me choose", "which is better", "suggest me a course", "which program", "i want to do", "parents want", "should i choose", "which course should", "better scope", "best course"}
    if any(sig in cleaned for sig in counselling_signals):
        return {
            "status": "ok",
            "message": "",
            "tag": "course_counselling",
            "confidence": 1.0,
            "similarity": 1.0,
            "method": "rule-based-keyword",
        }

    # Route failed / low marks eligibility queries
    failed_signals = {"failed", "fail", "compartment", "backlog", "did not pass", "low marks", "less than 50"}
    if any(sig in cleaned for sig in failed_signals):
        return {
            "status": "ok",
            "message": "",
            "tag": "failed_class12",
            "confidence": 1.0,
            "similarity": 1.0,
            "method": "rule-based-keyword",
        }

    brain_decision = _brain_select_intent(cleaned)
    if brain_decision and brain_decision["status"] == "ok":
        return {
            "status": "ok",
            "message": "",
            "tag": brain_decision["tag"],
            "confidence": brain_decision["confidence"],
            "similarity": brain_decision["similarity"],
            "method": brain_decision["method"],
        }

    predicted_tag, confidence = _predict_intent(cleaned)
    nearest_tag, similarity = _nearest_pattern_tag(cleaned)

    if nearest_tag and similarity >= HIGH_SIMILARITY_THRESHOLD:
        return {
            "status": "ok",
            "message": "",
            "tag": nearest_tag,
            "confidence": confidence,
            "similarity": similarity,
            "method": "pattern-high-similarity",
        }

    if confidence >= CONFIDENCE_THRESHOLD:
        return {
            "status": "ok",
            "message": "",
            "tag": predicted_tag,
            "confidence": confidence,
            "similarity": similarity,
            "method": "classifier-confidence",
        }

    if (
        nearest_tag
        and similarity >= LOW_CONFIDENCE_SIMILARITY_THRESHOLD
        and confidence >= LOW_CONFIDENCE_CLASSIFIER_MIN
    ):
        return {
            "status": "ok",
            "message": "",
            "tag": nearest_tag,
            "confidence": confidence,
            "similarity": similarity,
            "method": "pattern-fallback",
        }

    unsure_message = _uncertain_with_contact_message()

    return {
        "status": "uncertain",
        "message": unsure_message,
        "tag": None,
        "confidence": confidence,
        "similarity": similarity,
    }


def get_response(user_input):
    combined_predictions = _predict_intents_for_requirements(user_input)
    if combined_predictions:
        combined_response, _ = _compose_combined_response(combined_predictions)
        if combined_response:
            return combined_response

    prediction = predict_intent(user_input)

    if prediction["status"] != "ok":
        return prediction["message"]

    tag = prediction["tag"]
    response = _select_best_response_for_query(tag, user_input) or _find_response_by_tag(tag)
    if response:
        return _humanize_response(tag, user_input, response)

    return "I could not find a matching response. Please try a different question."


def _is_affirmative(text):
    cleaned = preprocess_text(text)
    if not cleaned:
        return False
    if cleaned in YES_WORDS:
        return True
    tokens = cleaned.split()
    return len(tokens) <= 2 and all(token in {"yes", "yess", "yeah", "ok", "okay"} for token in tokens)


def _is_greeting(text):
    cleaned = preprocess_text(text)
    return cleaned in GREETING_WORDS


def _is_goodbye(text):
    cleaned = preprocess_text(text)
    return cleaned in GOODBYE_WORDS


def _sample_question_by_tag(tag):
    for intent in data["intents"]:
        if intent.get("tag") == tag:
            patterns = intent.get("patterns", [])
            if patterns:
                return patterns[0]
    return None


def _course_key_from_tag(tag):
    return COURSE_TAG_TO_KEY.get(tag)


def _course_fee_response(course_key):
    return COURSE_FEE_OVERVIEWS.get(
        course_key,
        "The fee depends on the current admission cycle and the exact program. If you want, I can help you with the latest breakdown.",
    )


def _predict_next_questions(current_tag, limit=3):
    next_tags = INTENT_NEXT_TAGS.get(current_tag, [])
    suggestions = []
    for tag in next_tags:
        question = _sample_question_by_tag(tag)
        if question and "{course}" not in question and "business administration" not in question.lower():
            suggestions.append(question)
        if len(suggestions) >= limit:
            break
    return suggestions


def _fallback_suggestions(limit=5):
    return DEFAULT_GENERAL_SUGGESTIONS[:limit]


def _is_btech_deep_query(user_input, predicted_tag):
    cleaned = preprocess_text(user_input)
    if "btech" not in cleaned and "engineering" not in cleaned and "cse" not in cleaned:
        return False

    if predicted_tag in {"btech_details", "eligibility_criteria_btech", "btech_camparison"}:
        return True

    detail_cues = {
        "detail",
        "details",
        "complete",
        "full",
        "all",
        "everything",
        "specialization",
        "specializations",
        "types",
        "branches",
        "scope",
        "career",
        "roadmap",
    }

    return any(cue in cleaned for cue in detail_cues)


def _build_btech_targeted_response(user_input):
    cleaned = preprocess_text(user_input)
    if not any(keyword in cleaned for keyword in {"btech", "engineering", "cse", "ai ml", "aiml"}):
        return None, None

    # Don't hijack admission process queries
    if any(word in cleaned for word in {"admission", "apply", "application", "register", "registration", "document", "form"}):
        return None, None

    # Don't hijack faculty queries
    if any(word in cleaned for word in {"faculty", "professor", "teacher", "dean", "hod", "head of department", "teaches", "who teaches", "who is"}):
        return None, None

    if any(word in cleaned for word in {"eligibility", "eligible", "criteria", "qualification", "minimum"}):
        response = _select_best_response_for_query("eligibility_criteria_btech", user_input) or _find_response_by_tag("eligibility_criteria_btech")
        return response, [
            "What documents are needed for BTech admission?",
            "Is entrance score required for BTech admission?",
            "What is the next admission step for BTech?",
        ]

    if any(word in cleaned for word in {"fee", "fees", "cost", "tuition", "scholarship"}):
        response = _select_best_response_for_query("fees_general", user_input) or _find_response_by_tag("fees_general")
        return response, [
            "Can you share BTech scholarship options?",
            "Is hostel fee separate for BTech students?",
            "How can I get the latest BTech fee document?",
        ]

    if any(word in cleaned for word in {"specialization", "specializations", "type", "types", "branch", "branches"}):
        return BTECH_SPECIALIZATION_REPLY, [
            "How should I choose the right BTech specialization?",
            "Which BTech specialization has stronger career scope?",
            "Can you compare CSE and AI/ML tracks?",
        ]

    if any(word in cleaned for word in {"placement", "placements", "job", "package", "internship", "internships"}):
        response = _select_best_response_for_query("internship_info", user_input) or _find_response_by_tag("internship_info")
        return response, [
            "How are BTech placements overall?",
            "When do BTech internships usually start?",
            "How can I improve my internship chances?",
        ]

    if any(word in cleaned for word in {"difference", "compare", "comparison", "compairson", "comparision", "comparason", "regular", "collaboration"}):
        response = _select_best_response_for_query("btech_camparison", user_input) or _find_response_by_tag("btech_camparison")
        return response, [
            "What is the fee difference between these BTech tracks?",
            "Which track is better for placements?",
            "Can you compare the subject style?",
        ]

    if any(word in cleaned for word in {"syllabus", "curriculum", "subjects", "semester"}):
        response = _select_best_response_for_query("courses_curriculum", user_input) or _find_response_by_tag("courses_curriculum")
        return response, [
            "What are first-year BTech core subjects?",
            "Does BTech include projects and labs?",
            "How many semesters are there in BTech?",
        ]

    response = _select_best_response_for_query("btech_details", user_input) or _find_response_by_tag("btech_details")
    return response, BTECH_DETAIL_FOLLOWUPS


def _split_compound_requirements(user_input):
    cleaned_input = user_input.strip()
    if not cleaned_input:
        return []

    # Don't split if this looks like a comparison/between phrase or negation/failure context
    lowered = cleaned_input.lower()
    comparison_indicators = {"between", "versus", "vs", "difference", "compare", "comparison"}
    negation_indicators   = {"failed", "fail", "not pass", "cannot", "can't", "don't", "didn't", "without", "no ", "never"}
    if any(word in lowered for word in comparison_indicators | negation_indicators):
        return []

    split_pattern = "|".join(COMBINE_CONNECTOR_PATTERNS)
    parts = [segment.strip() for segment in re.split(split_pattern, cleaned_input, flags=re.IGNORECASE)]

    filtered_parts = []
    for segment in parts:
        cleaned_segment = preprocess_text(segment)
        if not cleaned_segment:
            continue
        token_count = len(cleaned_segment.split())
        if token_count >= 2 or cleaned_segment in SHORT_QUERY_ALLOWLIST:
            filtered_parts.append(segment)
    parts = filtered_parts

    # Only treat as compound when we genuinely got multiple meaningful parts.
    if len(parts) < 2:
        return []
    return parts[:3]


def _predict_intents_for_requirements(user_input):
    focused_query = _focus_user_query(user_input)
    segments = _split_compound_requirements(focused_query)
    if not segments:
        return []

    collected = []
    seen_tags = set()

    for segment in segments:
        prediction = predict_intent(segment)
        tag = prediction.get("tag")
        if prediction.get("status") != "ok" or not tag or tag in NON_COMBINABLE_TAGS:
            continue
        if tag in seen_tags:
            continue
        seen_tags.add(tag)
        collected.append({"segment": segment, "tag": tag, "prediction": prediction})

    if len(collected) < 2:
        return []
    return collected


def _compose_combined_response(collected_predictions):
    combined_lines = ["Here is a combined answer based on all parts of your question:"]
    suggestions = []

    for item in collected_predictions:
        tag = item["tag"]
        segment = item.get("segment", "")
        response = _select_best_response_for_query(tag, segment) or _find_response_by_tag(tag)
        if not response:
            continue
        response = _humanize_response(tag, segment, response)
        combined_lines.append(f"\n{_tag_title(tag)}: {response}")

        for suggestion in _predict_next_questions(tag):
            if suggestion not in suggestions:
                suggestions.append(suggestion)
            if len(suggestions) >= 5:
                break

    if len(combined_lines) == 1:
        return None, []

    return "\n".join(combined_lines), suggestions


def get_response_with_state(user_input, state=None):
    state = state or {}
    pending_followup = state.get("pending_followup")
    # Track the current topic course so follow-up questions stay in context
    current_course = state.get("current_course")   # e.g. "btech", "mba", "bba", "law"
    current_tag    = state.get("current_tag")       # last resolved intent tag

    # ── Greeting / Goodbye ────────────────────────────────────────────────
    if _is_greeting(user_input):
        response = _find_response_by_tag("greeting")
        if response:
            return response, {"pending_followup": None}, [
                "What is the admission process?",
                "What courses are offered?",
                "What is BTech eligibility?",
            ]

    if _is_goodbye(user_input) or any(w in user_input.lower() for w in {"thanks", "thank you", "thank u", "thankyou", "thx"}):
        response = _find_response_by_tag("goodbye")
        if response:
            return response, {"pending_followup": None}, [
                "What is the admission process?",
                "What courses are offered?",
            ]

    # ── Pending follow-up ─────────────────────────────────────────────────
    if pending_followup == FOLLOWUP_BTECH_COMPARISON and _is_affirmative(user_input):
        suggestions = [
            "What is the fee difference between regular and collaboration BTech?",
            "How are placements different in collaboration programs?",
            "What subjects are taught in first year BTech CSE?",
        ]
        return BTECH_COMPARISON_DETAILED_REPLY, {"pending_followup": None}, suggestions

    if isinstance(pending_followup, dict) and pending_followup.get("type") == COURSE_FEES_FOLLOWUP and _is_affirmative(user_input):
        course_key = pending_followup.get("course")
        suggestions = _predict_next_questions("fees_general")
        return _course_fee_response(course_key), {"pending_followup": None}, suggestions

    # ── Context injection: resolve vague follow-up queries using current_course ──
    cleaned_input = preprocess_text(user_input)

    VAGUE_ELIGIBILITY = {"eligibility", "eligible", "criteria", "qualification", "qualify", "am i eligible", "what is eligibility"}
    VAGUE_FEE        = {"fee", "fees", "cost", "how much", "tuition", "what is fee", "what are fees"}
    VAGUE_ADMISSION  = {"how to apply", "apply", "admission process", "how do i apply", "steps to apply"}

    if current_course and not any(course in cleaned_input for course in ["btech", "mba", "bba", "law", "hospitality", "design", "liberal"]):
        if any(term in cleaned_input for term in VAGUE_ELIGIBILITY):
            user_input = f"{user_input} for {current_course}"
        elif any(term in cleaned_input for term in VAGUE_FEE):
            user_input = f"{user_input} for {current_course}"
        elif any(term in cleaned_input for term in VAGUE_ADMISSION):
            user_input = f"{user_input} for {current_course}"

    # ── Context injection: percentage/score statements → direct eligibility verdict ──
    SCORE_PATTERNS = ["% in", "percent in", "marks in", "score in", "i have", "i got", "i scored"]
    if any(p in cleaned_input for p in SCORE_PATTERNS):
        # Try to extract a percentage number
        import re as _re
        pct_match = _re.search(r"(\d+(?:\.\d+)?)\s*%", user_input)
        pct = float(pct_match.group(1)) if pct_match else None

        course = current_course or ""
        if "pcm" in cleaned_input or "btech" in cleaned_input or course == "btech":
            if pct is not None:
                if pct >= 75:
                    verdict = f"With {pct}% in PCM, you have a strong profile for BTech at IILM. You should be eligible based on the general criteria. I recommend applying through https://iilm.edu/apply-now/"
                elif pct >= 60:
                    verdict = f"With {pct}% in PCM, you are likely eligible for BTech at IILM. Most engineering programs require 60%+ in PCM. Apply here: https://iilm.edu/apply-now/"
                else:
                    verdict = f"With {pct}% in PCM, eligibility may be borderline. I recommend contacting the admissions team directly to confirm: https://iilm.edu/contact/"
            else:
                verdict = "If you have completed 10+2 with Physics, Chemistry, and Mathematics (PCM) with the required percentage, you are likely eligible for BTech. Share your exact percentage and I can give a clearer verdict."
            return verdict, {"pending_followup": None, "current_course": "btech", "current_tag": "eligibility_criteria_btech"}, [
                "What is the BTech fee?", "Is there scholarship for BTech?", "How do I apply for BTech?"
            ]

        if "graduation" in cleaned_input or "mba" in cleaned_input or course == "mba":
            if pct is not None:
                if pct >= 50:
                    verdict = f"With {pct}% in graduation, you meet the standard MBA eligibility criteria at IILM. A good CAT/MAT/XAT score will further strengthen your application."
                else:
                    verdict = f"With {pct}% in graduation, eligibility may vary. I recommend checking directly with admissions: https://iilm.edu/contact/"
            else:
                verdict = "For MBA, IILM typically requires a bachelor's degree with at least 50% marks. Final-year students can also apply provisionally. Share your percentage for a more specific answer."
            return verdict, {"pending_followup": None, "current_course": "mba", "current_tag": "eligibility_criteria_mba"}, [
                "What is the MBA fee?", "What entrance exam do I need?", "Is scholarship available for MBA?"
            ]

    # ── Vague fee query in course context → resolve with course ──────────
    if current_course and any(term in cleaned_input for term in VAGUE_FEE):
        if not any(c in cleaned_input for c in ["btech", "mba", "bba", "law", "hospitality", "design", "liberal"]):
            course_fee_map = {
                "btech": "For BTech at IILM, the tuition fee typically ranges by campus and specialization. For the exact current fee breakdown, check: https://iilm.edu/admissions/ or contact +91-080 6590 5220.",
                "mba": "For MBA at IILM, fee varies by campus and intake. The total annual fee is listed on: https://iilm.edu/mba-admissions/ or contact admissions@iilm.edu.",
                "bba": "For BBA at IILM, the fee structure is listed on: https://iilm.edu/undergraduate-admissions/ — it varies by campus.",
                "law": "For Law programs at IILM, fees depend on whether you are enrolling in BA LLB (5-year) or LLB (3-year). Check: https://iilm.edu/admissions/",
            }
            if current_course in course_fee_map:
                return course_fee_map[current_course], {"pending_followup": None, "current_course": current_course, "current_tag": "fees_general"}, [
                    f"Is there scholarship for {current_course.upper()}?",
                    "How do I apply?",
                    "What are the hostel charges?",
                ]

    # ── "Tell me about MBA/BBA/Law" → route to course-specific overview ──
    TELL_ME_ABOUT_COURSES = {
        "mba": "eligibility_criteria_mba",
        "bba": "eligibility_criteria_bba",
        "btech": "btech_details",
        "law": "eligibility_criteria_law",
        "hospitality": "hospitality_info",
        "design": "design_fashion_info",
        "liberal arts": "liberal_arts_info",
    }
    if any(p in cleaned_input for p in {"tell me about", "details about", "about the", "tell me more"}):
        for course_kw, course_tag in TELL_ME_ABOUT_COURSES.items():
            if course_kw in cleaned_input:
                response = _select_best_response_for_query(course_tag, user_input) or _find_response_by_tag(course_tag)
                if response:
                    response = _humanize_response(course_tag, user_input, response)
                    new_state = {"pending_followup": None, "current_course": course_kw, "current_tag": course_tag}
                    return response, new_state, _predict_next_questions(course_tag) or _fallback_suggestions()

    # ── Entrance exam context question ────────────────────────────────────
    EXAM_SIGNALS = {"entrance exam", "entrance test", "which exam", "what exam", "exam needed", "exam required", "exam do i need"}
    if any(sig in cleaned_input for sig in EXAM_SIGNALS):
        course = current_course or ""
        if "mba" in course or "mba" in cleaned_input:
            return (
                "For MBA at IILM, the commonly accepted entrance exams are CAT, MAT, XAT, and CMAT. "
                "Direct admission based on academics is also possible. "
                "If you already have a score, share it and I can tell you if it qualifies.",
                {"pending_followup": None, "current_course": "mba", "current_tag": "eligibility_criteria_mba"},
                ["What is the MBA fee?", "Is scholarship available for MBA?", "How do I apply for MBA?"]
            )
        if "btech" in course or "btech" in cleaned_input:
            return (
                "For BTech at IILM, JEE Main score is considered but not always mandatory. "
                "IILM conducts its own admission process too. "
                "If you have a JEE score, share it and I can help you assess your chances.",
                {"pending_followup": None, "current_course": "btech", "current_tag": "eligibility_criteria_btech"},
                ["What is the BTech fee?", "Is there scholarship for BTech?", "How do I apply for BTech?"]
            )
        if "law" in course or "law" in cleaned_input:
            return (
                "For Law (BA LLB) at IILM, CLAT score is considered. "
                "Some students are also admitted based on merit. "
                "Check: https://iilm.edu/admissions/ for the latest criteria.",
                {"pending_followup": None, "current_course": "law", "current_tag": "eligibility_criteria_law"},
                ["What is the Law fee?", "Am I eligible for Law?", "How do I apply for Law?"]
            )

    # ── Combined intents ──────────────────────────────────────────────────
    combined_predictions = _predict_intents_for_requirements(user_input)
    if combined_predictions:
        combined_response, combined_suggestions = _compose_combined_response(combined_predictions)
        if combined_response:
            return combined_response, {"pending_followup": None, "current_course": current_course, "current_tag": current_tag}, combined_suggestions

    # ── Main intent prediction ────────────────────────────────────────────
    prediction = predict_intent(user_input)

    if prediction["status"] != "ok":
        return prediction["message"], state, _fallback_suggestions()

    tag = prediction["tag"]

    targeted_btech_response, targeted_btech_suggestions = _build_btech_targeted_response(user_input)
    if targeted_btech_response:
        new_state = {"pending_followup": None, "current_course": "btech", "current_tag": "btech_details"}
        return targeted_btech_response, new_state, targeted_btech_suggestions

    response = _select_best_response_for_query(tag, user_input) or _find_response_by_tag(tag)
    if not response:
        return "I could not find a matching response. Please try a different question.", state, _fallback_suggestions()

    response = _humanize_response(tag, user_input, response)

    # Deduplicate: remove double contact phone/email if it appears twice
    phone, email = _extract_admission_contact()
    if response.count(phone) > 1:
        last_idx = response.rfind(phone)
        sentence_start = response.rfind("You can contact", 0, last_idx)
        if sentence_start != -1:
            response = response[:sentence_start].strip()

    # ── Update state ──────────────────────────────────────────────────────
    new_state = {"pending_followup": None, "current_tag": tag}

    # Infer current_course from tag
    tag_to_course = {
        "btech_details": "btech", "eligibility_criteria_btech": "btech",
        "eligibility_criteria_mba": "mba", "eligibility_criteria_bba": "bba",
        "eligibility_criteria_law": "law", "hospitality_info": "hospitality",
        "design_fashion_info": "design", "liberal_arts_info": "liberal arts",
    }
    new_course = tag_to_course.get(tag, current_course)
    new_state["current_course"] = new_course

    if tag == "btech_camparison":
        new_state["pending_followup"] = FOLLOWUP_BTECH_COMPARISON

    course_key = _course_key_from_tag(tag)
    if course_key:
        followup_prompt = COURSE_FOLLOWUP_PROMPTS.get(course_key)
        if followup_prompt:
            response = f"{response}\n\n{followup_prompt}"
            new_state["pending_followup"] = {"type": COURSE_FEES_FOLLOWUP, "course": course_key}

    suggestions = _predict_next_questions(tag) or _fallback_suggestions(limit=3)

    return response, new_state, suggestions