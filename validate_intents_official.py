import json
import os
from copy import deepcopy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTENTS_PATH = os.path.join(BASE_DIR, "intents.json")

BANNED_TERMS = [
    "ai/ml",
    "ai ml",
    "cloud computing",
    "cybersecurity",
    "full stack",
    "data science",
    "analytics",
    "finance",
    "marketing",
    "human resources",
    "commerce",
    "product design",
    "ui/ux",
    "visual communication",
]

OFFICIAL_EXTRA_INTENTS = [
    {
        "tag": "liberal_arts_info",
        "patterns": [
            "What is Liberal Arts?",
            "Does IILM offer Liberal Arts?",
            "Can I apply for Liberal Arts?",
            "Tell me about Liberal Arts at IILM",
            "Is Liberal Arts available at IILM?",
        ],
        "responses": [
            "IILM offers Liberal Arts as part of its academic areas, with a focus on interdisciplinary learning and broad-based student development.",
            "If you are interested in Liberal Arts, I can help you compare it with other programs based on your background and career goals.",
        ],
        "context": "courses",
        "source": "iilm_website",
    },
    {
        "tag": "design_fashion_info",
        "patterns": [
            "What is Design & Fashion?",
            "Does IILM offer Design & Fashion?",
            "Can I apply for Design?",
            "Tell me about Design & Fashion at IILM",
            "Is Design available at IILM?",
        ],
        "responses": [
            "IILM includes Design & Fashion among its academic areas, with emphasis on creative learning, practical projects, and portfolio-based growth.",
            "If you are exploring Design & Fashion, I can also help you compare it with other programs and explain the admission path.",
        ],
        "context": "courses",
        "source": "iilm_website",
    },
    {
        "tag": "hospitality_info",
        "patterns": [
            "What is Hospitality?",
            "Does IILM offer Hospitality?",
            "Can I apply for Hospitality?",
            "Tell me about Hospitality at IILM",
            "Is Hospitality available at IILM?",
        ],
        "responses": [
            "IILM lists Hospitality among the undergraduate program areas on its official admissions pages. The focus is on practical learning and industry readiness.",
            "If you want, I can help you compare Hospitality with other IILM programs based on your academic background.",
        ],
        "context": "courses",
        "source": "iilm_website",
    },
]

RESPONSE_UPDATES = {
    "courses_offered": [
        "IILM offers programs in Management, Liberal Arts, Design & Fashion, Technology/Engineering, Law, and Hospitality. If you want, I can help you choose the best fit for your background.",
        "At IILM, the main program areas include Management, Liberal Arts, Design & Fashion, Technology, Law, Engineering, and Hospitality. Ask me about any one of them for a more specific explanation.",
        "IILM provides undergraduate and postgraduate options across its official study areas. Tell me your stream or degree level and I’ll narrow it down.",
    ],
    "btech_details": [
        "IILM’s engineering program focuses on core technical learning, projects, labs, and career-oriented skill building.",
        "The engineering degree is structured to build fundamentals first, then move toward practical projects and specialization choices.",
        "If you want, I can also explain how engineering compares with other IILM programs like Management or Law.",
    ],
    "btech_camparison": [
        "If you are comparing engineering options, think in terms of core fundamentals versus more applied learning tracks. I can help you decide based on your goals.",
        "A stronger fundamentals-focused engineering path is usually best if you want flexibility, while a more applied option is better if you want job-ready practical exposure.",
        "I can also break down the comparison by fees, placements, or subjects if you want the next level of detail.",
    ],
    "fees_general": [
        "Fees vary by program, campus, and current admission cycle. If you share the course name, I can give you the most relevant fee guidance.",
        "For the latest fee structure, it is best to check the official admission office or brochure for the exact program you are asking about.",
        "If you want, I can also explain the fee components separately, like tuition, hostel, transport, and other charges.",
    ],
    "placements_overview": [
        "IILM supports students with placements, internships, interview preparation, and career mentoring.",
        "Placement outcomes depend on the program, skills, and student preparation, so I can help you understand the career path for a specific course.",
        "If you want, I can also tell you what kind of preparation usually helps for interviews and internships.",
    ],
    "scholarship_info": [
        "IILM scholarships are mainly merit-based. Strong academics and valid entrance scores may improve your scholarship chances, depending on the program.",
        "Scholarship details change by admission cycle, so the exact amount and rules should be checked in the latest brochure or with the admission office.",
        "If you share your course and marks, I can tell you whether you are likely to get a scholarship.",
    ],
    "eligibility_criteria_mba": [
        "For MBA, IILM typically looks for a bachelor’s degree from a recognized university along with the required minimum score and applicable entrance-test criteria.",
        "Final-year students may also be eligible to apply, subject to official verification and the admission rules for the current cycle.",
        "If you want, I can also explain the MBA eligibility in simpler terms based on your background.",
    ],
    "eligibility_criteria_btech": [
        "For engineering, the usual requirement is 12th with the relevant science subjects and the minimum marks specified by the college.",
        "If you want, I can explain the engineering eligibility in relation to your current marks and subject combination.",
    ],
    "eligibility_criteria_bba": [
        "For BBA, the usual requirement is 12th from a recognized board with the minimum marks set by the college.",
        "If you want, I can also help you see whether BBA suits your background.",
    ],
    "eligibility_criteria_law": [
        "For Law, the requirements depend on whether you are applying after 12th or after graduation, along with the marks criteria for the current cycle.",
        "If you want, I can explain the Law eligibility in a simple way based on your current qualification.",
    ],
}


def should_drop_pattern(pattern):
    lowered = pattern.lower()
    return any(term in lowered for term in BANNED_TERMS)


def unique_append(items, candidate):
    candidate = candidate.strip()
    if candidate and candidate not in items:
        items.append(candidate)


def prune_intent_patterns(intent):
    patterns = intent.get("patterns", [])
    kept_patterns = [pattern for pattern in patterns if not should_drop_pattern(pattern)]
    intent["patterns"] = kept_patterns


def update_responses(intent):
    tag = intent.get("tag")
    if tag in RESPONSE_UPDATES:
        intent["responses"] = RESPONSE_UPDATES[tag]


def add_official_intents(data):
    existing_tags = {intent.get("tag") for intent in data.get("intents", [])}
    for extra_intent in OFFICIAL_EXTRA_INTENTS:
        if extra_intent["tag"] not in existing_tags:
            data["intents"].append(deepcopy(extra_intent))


def add_official_patterns(intent):
    tag = intent.get("tag")
    patterns = intent.setdefault("patterns", [])

    if tag == "courses_offered":
        for candidate in ["management", "MBA", "BBA", "engineering", "BTech", "law", "Liberal Arts", "Design & Fashion", "hospitality"]:
            unique_append(patterns, f"Does IILM offer {candidate}?")
            unique_append(patterns, f"Is {candidate} available at IILM?")
    elif tag == "fees_general":
        for candidate in ["management", "MBA", "BBA", "engineering", "BTech", "law", "Liberal Arts", "Design & Fashion", "hospitality"]:
            unique_append(patterns, f"What is the fee structure for {candidate}?")
            unique_append(patterns, f"How much are the fees for {candidate}?")
    elif tag.startswith("eligibility_criteria"):
        for candidate in ["management", "MBA", "BBA", "engineering", "BTech", "law", "Liberal Arts", "Design & Fashion", "hospitality"]:
            unique_append(patterns, f"What is eligibility for {candidate}?")
            unique_append(patterns, f"Am I eligible for {candidate}?")
    elif tag == "placements_overview":
        for candidate in ["management", "MBA", "BBA", "engineering", "BTech", "law", "Liberal Arts", "Design & Fashion", "hospitality"]:
            unique_append(patterns, f"What are the placement opportunities for {candidate}?")
    elif tag == "scholarship_info":
        for candidate in ["management", "MBA", "BBA", "engineering", "BTech", "law", "Liberal Arts", "Design & Fashion", "hospitality"]:
            unique_append(patterns, f"What scholarships are available for {candidate}?")
    elif tag == "admission_process":
        for candidate in ["management", "MBA", "BBA", "engineering", "BTech", "law", "Liberal Arts", "Design & Fashion", "hospitality"]:
            unique_append(patterns, f"What is the admission process for {candidate}?")


def main():
    with open(INTENTS_PATH, encoding="utf-8") as handle:
        data = json.load(handle)

    add_official_intents(data)

    for intent in data.get("intents", []):
        prune_intent_patterns(intent)
        update_responses(intent)
        add_official_patterns(intent)

    with open(INTENTS_PATH, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)

    pattern_total = sum(len(intent.get("patterns", [])) for intent in data["intents"])
    response_total = sum(len(intent.get("responses", [])) for intent in data["intents"])

    print("Official validation pass completed.")
    print(f"Intents: {len(data['intents'])}")
    print(f"Patterns: {pattern_total}")
    print(f"Responses: {response_total}")


if __name__ == "__main__":
    main()
