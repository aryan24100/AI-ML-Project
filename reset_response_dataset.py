import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INTENTS_PATH = BASE_DIR / "intents.json"

RESPONSES_BY_TAG = {
    "admission_process": [
        "Admission is simple: choose your program, submit the application form, upload required documents, and complete the admission review process.",
        "For exact steps and latest timeline, use the admissions page: https://iilm.edu/admissions/",
        "If you share your course, I can tell you the exact next step for that program."
    ],
    "eligibility_criteria_general": [
        "Eligibility depends on the program level (UG, PG, MBA, doctoral) and your academic background.",
        "Tell me the exact course name and I will give you the specific eligibility criteria only for that course.",
        "You can also verify official criteria here: https://iilm.edu/admissions/"
    ],
    "eligibility_criteria_btech": [
        "For BTech, eligibility is based on 10+2 qualification with required subject criteria as per current admission policy.",
        "If you share your stream and marks, I can tell you whether your profile is likely eligible.",
        "For the latest official rule set, check: https://iilm.edu/undergraduate-admissions/"
    ],
    "eligibility_criteria_mba": [
        "MBA eligibility is generally based on graduation status and current admission criteria for management programs.",
        "If you share your graduation background and entrance score (if any), I can give a precise guidance.",
        "Official page: https://iilm.edu/mba-admissions/"
    ],
    "eligibility_criteria_bba": [
        "BBA eligibility is based on 10+2 completion as per current UG admission rules.",
        "If you share your stream and marks, I can answer in a yes/no format for your profile.",
        "Official UG admissions page: https://iilm.edu/undergraduate-admissions/"
    ],
    "eligibility_criteria_law": [
        "Law program eligibility depends on your qualification level and current admission policy for legal studies.",
        "If you share whether you are applying after 12th or graduation, I can provide the exact route.",
        "Official admissions page: https://iilm.edu/admissions/"
    ],
    "courses_offered": [
        "IILM offers programs across undergraduate, postgraduate, MBA, and doctoral categories.",
        "To see the latest official list, check: https://iilm.edu/programmes/",
        "If you tell me your background (after 12th or after graduation), I can suggest the best-fit options only."
    ],
    "btech_details": [
        "BTech is designed as a 4-year program with core subjects, labs, projects, and industry-oriented learning.",
        "If you want a specific detail, ask one: eligibility, fees, specializations, placements, or curriculum.",
        "I can answer each BTech topic one by one so it stays clear and not overloaded."
    ],
    "btech_camparison": [
        "Regular BTech focuses on strong core fundamentals, while collaboration-oriented tracks are usually more industry-aligned.",
        "The practical difference is usually in specialization depth, exposure format, and overall fee structure.",
        "If you want, I can compare only one dimension at a time: fees, subjects, or placements."
    ],
    "placements_overview": [
        "Placement support is provided through training, interview preparation, and employer engagement.",
        "Actual outcomes depend on program, skills, projects, and student performance.",
        "Official placement info is available here: https://iilm.edu/placements/"
    ],
    "contact_info": [
        "You can contact the admission team directly through the official contact page: https://iilm.edu/contact/",
        "For application help, you can also use: https://iilm.edu/apply-now/",
        "Admission support email and helpline are listed on the official contact/admissions pages."
    ],
    "scholarship_info": [
        "Scholarship options are admission-cycle specific and program dependent.",
        "For official scholarship document, check the admissions section and scholarship PDF links on: https://iilm.edu/admissions/",
        "If you share your marks/score, I can tell you the likely scholarship path to check first."
    ],
    "campus_location": [
        "IILM has campuses including Gurugram and Greater Noida, along with other centers listed on the official site.",
        "Campus pages are listed here: https://iilm.edu/admissions/",
        "Tell me which campus you need and I will give only that location guidance."
    ],
    "courses_curriculum": [
        "Curriculum is program-specific and usually includes core papers, electives, projects, and practical components.",
        "If you share the exact course name, I can give a focused curriculum summary for that course only.",
        "Official program pages: https://iilm.edu/programmes/"
    ],
    "hostel_info": [
        "Hostel details vary by campus and room type.",
        "For exact availability and latest charges, check with admissions for your campus.",
        "If you share your preferred campus, I can narrow down what to ask first."
    ],
    "mess_info": [
        "Mess facilities are typically linked with hostel/campus accommodation policies.",
        "Food options and timings can vary by campus and session.",
        "Tell me your campus and I will give the exact query list to ask admissions."
    ],
    "transportation_info": [
        "Transport availability depends on campus routes and current shuttle/bus schedule.",
        "For route-wise confirmation, admissions/support desk is the most accurate source.",
        "If you share your commute area, I can suggest the right transport question to ask."
    ],
    "counselling_support": [
        "Yes, admission guidance/counselling support is available to help you choose the right program.",
        "Share your background and goals, and I can prepare a focused shortlist before you contact admissions.",
        "Official contact page: https://iilm.edu/contact/"
    ],
    "sports_facilities": [
        "Sports and student activity facilities are part of campus life.",
        "Exact facilities can differ by campus.",
        "Tell me your campus preference and I will provide a focused checklist."
    ],
    "faq_general": [
        "IILM provides career-focused programs with admissions, academics, and student support across multiple campuses.",
        "If you ask a specific topic (admission, fees, placements, hostel, scholarship), I can give a precise answer quickly.",
        "For official updates, use: https://iilm.edu/"
    ],
    "events_and_clubs": [
        "Yes, student clubs and events are part of campus life.",
        "Event updates are available on the official events page: https://iilm.edu/events/",
        "If you want, I can suggest what to ask to evaluate student culture before admission."
    ],
    "international_programs": [
        "International opportunities can include global exposure pathways based on program structure.",
        "Availability may vary by course and intake.",
        "If you share your course interest, I can give focused guidance on this topic."
    ],
    "faculty_info": [
        "Faculty includes academic and industry-oriented teaching support depending on program.",
        "For quality evaluation, you can check curriculum, projects, and placement outcomes together.",
        "Tell me your target course and I will give a focused faculty-related checklist."
    ],
    "canteen_food": [
        "Food and canteen facilities are available as part of campus services.",
        "Exact options and timings can vary by campus.",
        "If you share campus preference, I can provide focused questions to confirm."
    ],
    "fallback": [
        "I did not fully understand that. Please ask in one clear line, for example: BTech eligibility, fee structure, placements, or contact details."
    ],
    "greeting": [
        "Hello! Ask me one specific question and I will give a direct answer."
    ],
    "goodbye": [
        "Thanks. If needed, come back with a specific query and I will keep it short and clear."
    ],
    "fees_general": [
        "Fee structure depends on program, campus, and admission cycle.",
        "For official fee document, use the admissions page resources: https://iilm.edu/admissions/",
        "Tell me your exact course and campus, and I will guide you to the right fee breakdown."
    ],
    "internship_info": [
        "Yes, internship support is part of student development and career preparation.",
        "Internship outcomes depend on your program, skills, projects, and profile quality.",
        "If you share your course, I can tell you the internship path typically followed by students."
    ],
    "college_timings": [
        "Class and office timings can vary by program and campus.",
        "For exact schedule, verify with admissions or your target department.",
        "If you share the campus/program, I can suggest the exact timing questions to ask."
    ],
    "website_navigation": [
        "Use these official links: Admissions https://iilm.edu/admissions/ | Apply https://iilm.edu/apply-now/ | Contact https://iilm.edu/contact/ | Programs https://iilm.edu/programmes/",
        "Tell me what you want to find and I will give the exact page directly.",
        "If a link is not loading, I can suggest an alternate official page."
    ],
    "facilities": [
        "Campus facilities vary by location and program setup.",
        "Common student services include academic support and campus life infrastructure.",
        "Tell me your preferred campus and I will give a focused facilities checklist."
    ],
    "liberal_arts_info": [
        "Liberal Arts options are available under relevant program offerings.",
        "For current structure and admissions details, check: https://iilm.edu/programmes/",
        "If you share your interests, I can guide the right Liberal Arts path to explore."
    ],
    "design_fashion_info": [
        "Design and related creative program details are available on the official program pages.",
        "Please check: https://iilm.edu/programmes/ for latest structure and intake information.",
        "If you tell me your design interest area, I can give focused next questions."
    ],
    "hospitality_info": [
        "Hospitality-related program information is available through the official program/admission pages.",
        "Use: https://iilm.edu/programmes/ and https://iilm.edu/admissions/ for latest updates.",
        "If you want, I can help you compare hospitality with other options based on career goals."
    ],
}


def main():
    with INTENTS_PATH.open(encoding="utf-8") as f:
        data = json.load(f)

    tags_in_file = {intent.get("tag") for intent in data.get("intents", [])}
    tags_in_map = set(RESPONSES_BY_TAG)

    missing_in_map = sorted(tags_in_file - tags_in_map)
    extra_in_map = sorted(tags_in_map - tags_in_file)

    if missing_in_map:
        raise ValueError(f"Missing response templates for tags: {missing_in_map}")

    if extra_in_map:
        print(f"Warning: templates include unknown tags and will be ignored: {extra_in_map}")

    for intent in data.get("intents", []):
        tag = intent.get("tag")
        intent["responses"] = RESPONSES_BY_TAG[tag]

    with INTENTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("Responses dataset reset successfully.")
    print(f"Updated intents: {len(data.get('intents', []))}")
    print(f"Total response lines: {sum(len(i.get('responses', [])) for i in data.get('intents', []))}")


if __name__ == "__main__":
    main()
