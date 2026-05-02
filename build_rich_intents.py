import json
import os
import re
from collections import Counter

from nlp_utils import preprocess_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTENTS_PATH = os.path.join(BASE_DIR, "intents.json")

COURSE_ALIASES = {
    "btech": [
        "BTech",
        "B.Tech",
        "engineering",
        "computer science",
        "CSE",
        "AI/ML",
        "cloud computing",
        "cybersecurity",
        "full stack",
    ],
    "mba": [
        "MBA",
        "management",
        "business administration",
        "finance",
        "marketing",
        "human resources",
    ],
    "bba": [
        "BBA",
        "business administration",
        "commerce",
        "management",
    ],
    "law": [
        "law",
        "LLB",
        "BA LLB",
        "legal studies",
        "CLAT",
    ],
    "design": [
        "design",
        "B.Des",
        "product design",
        "UI/UX",
        "visual communication",
    ],
    "science": [
        "BSc",
        "science",
        "data science",
        "analytics",
    ],
}

TOPIC_TEMPLATES = {
    "admission_process": [
        "what is the admission process for {course}",
        "how do i apply for {course} at iilm",
        "can i take admission in {course} online",
        "can i apply offline for {course}",
        "what documents are needed for admission in {course}",
        "is admission for {course} merit based",
        "is entrance exam needed for {course} admission",
        "how to register for {course}",
        "when does {course} admission start",
        "can i get direct admission in {course}",
        "is the application form available for {course}",
        "what is the step by step process for {course} admission",
    ],
    "eligibility_criteria_general": [
        "what is eligibility for {course}",
        "am i eligible for {course}",
        "who can apply for {course}",
        "what marks are needed for {course}",
        "what is the minimum qualification for {course}",
        "what are the requirements for {course}",
        "can i apply for {course} after 12th",
        "can i apply for {course} after graduation",
        "what subjects are needed for {course}",
    ],
    "eligibility_criteria_btech": [
        "what is eligibility for BTech",
        "BTech eligibility criteria",
        "can i apply for BTech after 12th",
        "what marks are required for BTech",
        "do i need pcm for BTech",
        "is jee required for BTech",
        "can i get BTech without jee",
        "what subjects are needed for engineering",
        "eligibility for BTech CSE",
        "eligibility for AI ML BTech",
        "what qualification is needed for BTech",
        "who can apply for BTech at IILM",
    ],
    "eligibility_criteria_mba": [
        "what is eligibility for MBA",
        "MBA eligibility criteria",
        "can i apply for MBA after graduation",
        "what marks are required for MBA",
        "do i need cat for MBA",
        "can i get MBA without entrance exam",
        "eligibility for MBA program",
        "who can apply for MBA at IILM",
        "can final year students apply for MBA",
    ],
    "eligibility_criteria_bba": [
        "what is eligibility for BBA",
        "BBA eligibility criteria",
        "can i do BBA after 12th",
        "what marks are needed for BBA",
        "can science students apply for BBA",
        "can commerce students apply for BBA",
        "can arts students apply for BBA",
        "who can apply for BBA at IILM",
    ],
    "eligibility_criteria_law": [
        "what is eligibility for law",
        "law eligibility criteria",
        "can i apply for BA LLB after 12th",
        "do i need clat for law admission",
        "what marks are required for law",
        "who can apply for law at IILM",
        "can i do llb after graduation",
        "eligibility for BA LLB",
    ],
    "courses_offered": [
        "what courses does IILM offer",
        "which programs are available at IILM",
        "does IILM offer {course}",
        "is {course} available at IILM",
        "what can i study at IILM",
        "which course should i choose after 12th",
        "which course should i choose after graduation",
        "do you have engineering and management programs",
        "tell me the list of courses",
        "what streams are offered at IILM",
    ],
    "fees_general": [
        "what is the fee structure for {course}",
        "how much are the fees for {course}",
        "what is the annual fee for {course}",
        "tell me the total cost of {course}",
        "is hostel fee included in {course} fees",
        "what is the semester fee for {course}",
        "can you give a fee breakdown for {course}",
        "how much money is needed for {course}",
        "what is the overall expense for {course}",
    ],
    "placements_overview": [
        "what are the placement opportunities for {course}",
        "does IILM provide placements for {course}",
        "which companies hire {course} students",
        "what is the average package for {course}",
        "what is the highest package for {course}",
        "how good are placements at IILM",
        "is internship support available for {course}",
        "placement record for {course}",
        "do students get jobs after {course}",
    ],
    "internship_info": [
        "do you provide internships for {course}",
        "when do {course} students get internships",
        "is internship compulsory for {course}",
        "how can i get internship support for {course}",
        "do companies offer internships to {course} students",
        "can i get an internship through college for {course}",
        "how internship works for {course} students",
    ],
    "courses_curriculum": [
        "what is the syllabus for {course}",
        "what subjects are taught in {course}",
        "tell me the curriculum for {course}",
        "what is the course structure of {course}",
        "what are the first year subjects for {course}",
        "does {course} include projects and labs",
        "how many semesters are in {course}",
    ],
    "scholarship_info": [
        "what scholarships are available for {course}",
        "how can i get scholarship for {course}",
        "is scholarship available for meritorious students",
        "can entrance scores help for scholarship in {course}",
        "what are the scholarship criteria for {course}",
        "does IILM give merit scholarship",
    ],
    "hostel_info": [
        "is hostel available for {course} students",
        "what are the hostel facilities",
        "how much is the hostel fee",
        "is hostel separate from tuition fee",
        "what type of rooms are available in hostel",
        "is hostel safe and comfortable",
        "can girls and boys stay in hostel",
    ],
    "transportation_info": [
        "is transportation available for students",
        "what are the bus routes",
        "does the college have transport facility",
        "how much is the transport fee",
        "is bus service available from campus",
        "what are the commuting options",
    ],
    "campus_location": [
        "where is the campus located",
        "what is the address of IILM",
        "can you tell me the campus location",
        "how do i reach the campus",
        "which campus is in greater noida",
        "where is the college situated",
    ],
    "college_timings": [
        "what time do classes start",
        "what are the college working hours",
        "when is the office open",
        "what are the class timings",
        "what are the admission office hours",
        "what is the daily schedule for students",
    ],
    "website_navigation": [
        "where can i find the admission form",
        "how to find the fee page on the website",
        "where is the scholarship section",
        "where can i check programs online",
        "how to navigate the official website",
        "where is the contact page on the website",
    ],
    "faculty_info": [
        "who teaches in {course}",
        "are the faculty experienced",
        "how are the teachers at IILM",
        "what is the faculty profile",
        "does the college have industry experts",
        "can i know about the professors",
    ],
    "facilities": [
        "what facilities are available on campus",
        "what student facilities does the college provide",
        "are there cafes and stores on campus",
        "what basic facilities are inside campus",
        "what are the daily life facilities",
        "does the campus have an atm and store",
    ],
    "faq_general": [
        "can you tell me about IILM in general",
        "give me a quick overview of the college",
        "what should i know before joining IILM",
        "tell me the main highlights of the college",
        "what makes IILM different",
    ],
    "events_and_clubs": [
        "what clubs are available on campus",
        "are there student events and activities",
        "what extracurricular activities are available",
        "does the college have cultural clubs",
        "what events happen in college",
    ],
    "international_programs": [
        "are there international programs",
        "does IILM have global exchange options",
        "can students study abroad through IILM",
        "are there international collaborations",
        "what global opportunities are available",
    ],
    "mess_info": [
        "is mess available in hostel",
        "what food is available in mess",
        "how is the mess facility",
        "is vegetarian food available",
        "what are the mess timings",
    ],
    "sports_facilities": [
        "what sports facilities are available",
        "does the campus have sports ground",
        "are indoor sports available",
        "can students play basketball or cricket",
        "what fitness facilities are there",
    ],
    "counselling_support": [
        "do you provide counselling support",
        "can i get help choosing a course",
        "is admission counselling available",
        "can someone guide me based on my background",
        "do you help students with career choice",
    ],
}

RESPONSE_UPDATES = {
    "fees_general": [
        "For most programs, fees depend on the course, specialization, and whether you choose a regular or industry-collaboration track. If you share the program name, I can narrow it down for you.",
        "If you want a complete cost picture, remember that tuition, hostel, transport, and other academic charges may be separate. Tell me the exact course and I’ll explain it clearly.",
    ],
    "courses_offered": [
        "IILM offers programs across engineering, management, law, design, liberal arts, science, and more. If you want, I can suggest the best option based on your background.",
        "For students after 12th, popular choices include B.Tech, BBA, and BA LL.B. For graduates, MBA and other postgraduate programs are available.",
    ],
    "hostel_info": [
        "Hostel availability usually depends on room type and campus policy. If you want, I can also explain the difference between hostel charges and tuition fees.",
        "Hostel facilities generally include room accommodation, security, and basic student support. Let me know if you want a fuller overview of campus living.",
    ],
    "campus_location": [
        "IILM has campuses in Greater Noida, Gurugram, and other locations. If you want directions, I can help you with the campus you are planning to visit.",
        "The main admission support for this chatbot is centered on the Greater Noida campus. If needed, I can also share the general visit guidance.",
    ],
    "faculty_info": [
        "IILM faculty usually includes experienced academicians and industry professionals. If you want, I can also tell you how that helps with placements and projects.",
        "Faculty support is designed to be practical and student-friendly, especially for project work, internships, and exam preparation.",
    ],
    "website_navigation": [
        "On the official website, admissions, programs, fees, and contact details are usually grouped in clear menu sections. If you want, I can point you to the exact section name.",
        "The website is structured so that application, fee, and program pages can be found quickly from the main menu.",
    ],
    "faq_general": [
        "IILM is positioned as a career-focused university with admissions, programs, placements, and student support organized around practical learning.",
    ],
    "events_and_clubs": [
        "Student clubs and events are part of the college experience and usually support leadership, teamwork, and communication skills.",
    ],
    "international_programs": [
        "International exposure can include collaborations, exchange opportunities, and global academic pathways depending on the program.",
    ],
    "mess_info": [
        "Mess facilities are usually separate from academic fees and may vary by campus and accommodation type.",
    ],
    "sports_facilities": [
        "Sports and recreation facilities are meant to support both fitness and campus life. If you want, I can also explain the student activity side of the campus.",
    ],
    "counselling_support": [
        "Counselling support can help you compare programs based on your stream, interests, and career goals.",
    ],
}

DEFAULT_PATTERN_CAP = 60
INTENT_PATTERN_CAPS = {
    "admission_process": 90,
    "eligibility_criteria_general": 80,
    "fees_general": 80,
    "courses_offered": 75,
    "placements_overview": 65,
    "scholarship_info": 65,
}

LOW_SIGNAL_PATTERNS = {
    "admission",
    "apply",
    "fees",
    "hostel",
    "placement",
    "placements",
    "eligibility",
    "courses",
}

STOP_WORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "am",
    "i",
    "me",
    "my",
    "you",
    "your",
    "for",
    "to",
    "of",
    "in",
    "on",
    "at",
    "and",
    "or",
    "with",
    "about",
    "can",
    "could",
    "would",
    "please",
    "tell",
    "give",
    "explain",
    "do",
    "does",
    "did",
    "any",
    "there",
    "this",
    "that",
}


def add_unique(items, candidate):
    candidate = candidate.strip()
    if candidate and candidate not in items:
        items.append(candidate)


def _normalize_pattern(pattern):
    return preprocess_text(pattern or "")


def _signature_tokens(pattern):
    cleaned = _normalize_pattern(pattern)
    return [token for token in cleaned.split() if token not in STOP_WORDS]


def _quality_score(pattern):
    cleaned = _normalize_pattern(pattern)
    tokens = cleaned.split()
    if not tokens:
        return -100.0

    score = 0.0
    token_count = len(tokens)
    unique_tokens = len(set(tokens))

    if 3 <= token_count <= 12:
        score += 3.0
    elif token_count == 2:
        score += 1.0
    else:
        score -= 1.5

    if any(word in cleaned for word in {"what", "how", "when", "where", "who", "which", "can"}):
        score += 1.0

    if "iilm" in cleaned:
        score += 0.5

    score += min(unique_tokens, 12) * 0.2
    return score


def _is_low_signal(pattern):
    cleaned = _normalize_pattern(pattern)
    if not cleaned:
        return True

    if cleaned in LOW_SIGNAL_PATTERNS:
        return True

    # Drop tiny patterns like "admission?" or "apply?" that hurt intent quality.
    if len(cleaned.split()) == 1 and cleaned not in {"hello", "hi", "bye", "goodbye"}:
        return True

    if len(cleaned.split()) == 2 and all(token in LOW_SIGNAL_PATTERNS for token in cleaned.split()):
        return True

    return False


def _is_near_duplicate(candidate_pattern, selected_patterns, jaccard_threshold=0.84):
    candidate_norm = _normalize_pattern(candidate_pattern)
    candidate_tokens = set(_signature_tokens(candidate_pattern))

    for existing in selected_patterns:
        existing_norm = _normalize_pattern(existing)
        if candidate_norm == existing_norm:
            return True

        existing_tokens = set(_signature_tokens(existing))
        if not candidate_tokens or not existing_tokens:
            continue

        intersection = len(candidate_tokens & existing_tokens)
        union = len(candidate_tokens | existing_tokens)
        if union == 0:
            continue

        jaccard = intersection / union
        containment = intersection / min(len(candidate_tokens), len(existing_tokens))

        if jaccard >= jaccard_threshold:
            return True
        if containment >= 0.9 and abs(len(candidate_tokens) - len(existing_tokens)) <= 2:
            return True

    return False


def _prune_patterns(patterns, cap):
    cleaned_candidates = []
    seen_norm = set()

    for pattern in patterns:
        normalized = _normalize_pattern(pattern)
        if not normalized or normalized in seen_norm:
            continue
        seen_norm.add(normalized)
        if _is_low_signal(pattern):
            continue
        cleaned_candidates.append(pattern.strip())

    cleaned_candidates.sort(key=lambda p: (_quality_score(p), len(_normalize_pattern(p))), reverse=True)

    selected = []
    for candidate in cleaned_candidates:
        if _is_near_duplicate(candidate, selected):
            continue
        selected.append(candidate)
        if len(selected) >= cap:
            break

    # Ensure every intent keeps at least a small usable set.
    if len(selected) < 12:
        for candidate in cleaned_candidates:
            if candidate in selected:
                continue
            selected.append(candidate)
            if len(selected) >= min(cap, 12):
                break

    return selected


def _prune_dataset_patterns(data):
    removed = 0
    for intent in data.get("intents", []):
        tag = intent.get("tag", "")
        cap = INTENT_PATTERN_CAPS.get(tag, DEFAULT_PATTERN_CAP)
        patterns = intent.get("patterns", [])
        pruned = _prune_patterns(patterns, cap)
        removed += max(0, len(patterns) - len(pruned))
        intent["patterns"] = pruned
    return removed


def course_forms(course_key):
    aliases = COURSE_ALIASES.get(course_key, [course_key])
    readable = [course_key.upper()] + aliases
    return readable


def build_course_patterns(tag, course_key):
    generated = []
    for template in TOPIC_TEMPLATES.get(tag, []):
        if "{course}" in template:
            for course_name in course_forms(course_key):
                add_unique(generated, template.format(course=course_name))
        else:
            add_unique(generated, template)
    return generated


def add_generic_followups(intent_tag, patterns):
    if intent_tag == "courses_offered":
        for course_key in COURSE_ALIASES:
            for course_name in course_forms(course_key):
                add_unique(patterns, f"does IILM offer {course_name}")
                add_unique(patterns, f"is {course_name} available")
                add_unique(patterns, f"tell me about {course_name} at IILM")
    elif intent_tag == "fees_general":
        for course_key in COURSE_ALIASES:
            for course_name in course_forms(course_key):
                add_unique(patterns, f"what is the fee for {course_name}")
                add_unique(patterns, f"how much does {course_name} cost")
                add_unique(patterns, f"can you tell me the fees of {course_name}")
    elif intent_tag.startswith("eligibility_criteria"):
        for course_key in COURSE_ALIASES:
            for course_name in course_forms(course_key):
                add_unique(patterns, f"am i eligible for {course_name}")
                add_unique(patterns, f"what is the eligibility for {course_name}")
                add_unique(patterns, f"who can apply for {course_name}")
    elif intent_tag == "placements_overview":
        for course_key in COURSE_ALIASES:
            for course_name in course_forms(course_key):
                add_unique(patterns, f"what are the placements for {course_name}")
                add_unique(patterns, f"placement scope after {course_name}")
    elif intent_tag == "scholarship_info":
        for course_key in COURSE_ALIASES:
            for course_name in course_forms(course_key):
                add_unique(patterns, f"scholarship for {course_name}")
                add_unique(patterns, f"can i get scholarship for {course_name}")


def enrich_intents(data):
    total_added_patterns = 0
    total_added_responses = 0

    for intent in data.get("intents", []):
        tag = intent.get("tag", "")
        patterns = intent.setdefault("patterns", [])
        responses = intent.setdefault("responses", [])

        before_patterns = len(patterns)
        before_responses = len(responses)

        if tag in TOPIC_TEMPLATES:
            for course_key in COURSE_ALIASES:
                for pattern in build_course_patterns(tag, course_key):
                    add_unique(patterns, pattern)

            if tag in {"courses_offered", "fees_general", "placements_overview", "scholarship_info"}:
                add_generic_followups(tag, patterns)
        elif tag in {"admission_process", "faq_general", "events_and_clubs", "international_programs", "college_timings", "website_navigation", "facilities", "campus_location", "hostel_info", "mess_info", "transportation_info", "counselling_support", "sports_facilities", "faculty_info", "internship_info", "courses_curriculum"}:
            for course_key in COURSE_ALIASES:
                for course_name in course_forms(course_key):
                    if tag == "internship_info":
                        add_unique(patterns, f"do you provide internships for {course_name}")
                        add_unique(patterns, f"when do {course_name} students get internships")
                    elif tag == "courses_curriculum":
                        add_unique(patterns, f"what subjects are taught in {course_name}")
                        add_unique(patterns, f"what is the curriculum of {course_name}")
                    elif tag == "faculty_info":
                        add_unique(patterns, f"who teaches {course_name}")
                    elif tag == "hostel_info":
                        add_unique(patterns, f"is hostel available for {course_name} students")
                    elif tag == "campus_location":
                        add_unique(patterns, f"where is the {course_name} campus")
                    elif tag == "website_navigation":
                        add_unique(patterns, f"where can i find {course_name} details on the website")
                    elif tag == "counselling_support":
                        add_unique(patterns, f"can you help me choose {course_name}")
                    elif tag == "admission_process":
                        add_unique(patterns, f"how do i apply for {course_name}")
                    elif tag == "faq_general":
                        add_unique(patterns, f"tell me about {course_name} at iilm")
                    elif tag == "events_and_clubs":
                        add_unique(patterns, f"are there clubs for {course_name} students")
                    elif tag == "international_programs":
                        add_unique(patterns, f"does {course_name} have global opportunities")
                    elif tag == "college_timings":
                        add_unique(patterns, f"what are the class timings for {course_name}")
                    elif tag == "facilities":
                        add_unique(patterns, f"what facilities do {course_name} students get")
                    elif tag == "mess_info":
                        add_unique(patterns, f"is mess available for {course_name} students")
                    elif tag == "transportation_info":
                        add_unique(patterns, f"is transport available for {course_name} students")
                    elif tag == "sports_facilities":
                        add_unique(patterns, f"what sports are available for {course_name} students")
                    elif tag == "internship_info":
                        add_unique(patterns, f"do {course_name} students get internships")
                    elif tag == "scholarship_info":
                        add_unique(patterns, f"is scholarship available for {course_name}")

        for response in RESPONSE_UPDATES.get(tag, []):
            add_unique(responses, response)

        total_added_patterns += len(patterns) - before_patterns
        total_added_responses += len(responses) - before_responses

    return total_added_patterns, total_added_responses


def main():
    with open(INTENTS_PATH, encoding="utf-8") as handle:
        data = json.load(handle)

    total_added_patterns, total_added_responses = enrich_intents(data)
    removed_patterns = _prune_dataset_patterns(data)

    with open(INTENTS_PATH, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)

    pattern_total = sum(len(intent.get("patterns", [])) for intent in data["intents"])
    response_total = sum(len(intent.get("responses", [])) for intent in data["intents"])
    tag_counts = Counter(intent.get("tag", "") for intent in data["intents"])

    print("Rich intent dataset written successfully.")
    print(f"Total intents: {len(data['intents'])}")
    print(f"Total patterns: {pattern_total}")
    print(f"Total responses: {response_total}")
    print(f"Added patterns: {total_added_patterns}")
    print(f"Added responses: {total_added_responses}")
    print(f"Removed redundant patterns: {removed_patterns}")
    print(f"Most common intent count: {tag_counts.most_common(1)[0] if tag_counts else ('n/a', 0)}")


if __name__ == "__main__":
    main()
