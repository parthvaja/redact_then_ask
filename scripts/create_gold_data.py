#!/usr/bin/env python3
"""Create synthetic sample documents and gold annotation data.

Writes:
  data/samples/<doc_type>_01.txt  (5 files, set 1)
  data/samples/<doc_type>_02.txt  (5 files, set 2)
  data/gold/handlabel.jsonl       (10 GoldDoc entries)
  data/gold/qa_template.jsonl     (100 QA pairs)

Every span offset is verified programmatically before writing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = ROOT / "data" / "samples"
GOLD_DIR = ROOT / "data" / "gold"

# ---------------------------------------------------------------------------
# Document texts
# ---------------------------------------------------------------------------

DOCUMENTS: dict[str, dict] = {}

# ---- medical_01 ----------------------------------------------------------
DOCUMENTS["medical_01"] = {
    "doc_type": "medical",
    "text": (
        "DISCHARGE SUMMARY\n"
        "Patient: Margaret Ellen Sullivan  DOB: 03/15/1958  Age: 66\n"
        "MRN: MRN-7842193  Phone: (410) 555-0173\n"
        "Address: 2847 Windmill Lane, Towson, MD 21204\n"
        "Attending Physician: Dr. Samuel R. Okonkwo\n"
        "\n"
        "Date of Admission: 09/02/2024  Date of Discharge: 09/08/2024\n"
        "\n"
        "Diagnosis: Acute exacerbation of chronic obstructive pulmonary disease "
        "(COPD) with right lower lobe pneumonia.\n"
        "\n"
        "Hospital Course: The patient was admitted through the emergency "
        "department with worsening dyspnea and productive cough. Chest X-ray "
        "confirmed right lower lobe consolidation. She was started on IV "
        "ceftriaxone and azithromycin. Supplemental oxygen was provided via "
        "nasal cannula at 2L/min. Symptoms improved steadily over the course "
        "of six days.\n"
        "\n"
        "Discharge Medications: Prednisone 40mg daily taper, Albuterol inhaler "
        "2 puffs q4h PRN, Azithromycin 250mg PO daily x3 days."
    ),
}

# ---- medical_02 ----------------------------------------------------------
DOCUMENTS["medical_02"] = {
    "doc_type": "medical",
    "text": (
        "OPERATIVE NOTE\n"
        "Patient: James Francis Dellacroce  DOB: 11/27/1945  Age: 80\n"
        "MRN: MRN-3310567  Phone: (301) 555-0298\n"
        "Address: 519 Cedar Park Drive, Silver Spring, MD 20910\n"
        "Surgeon: Dr. Priya Anand Mehta\n"
        "\n"
        "Date of Procedure: 10/14/2024\n"
        "\n"
        "Preoperative Diagnosis: Right inguinal hernia, recurrent.\n"
        "Postoperative Diagnosis: Same.\n"
        "\n"
        "Procedure Performed: Open right inguinal hernia repair with mesh "
        "(Lichtenstein technique).\n"
        "\n"
        "Findings: A large indirect hernia sac was identified extending into "
        "the scrotum. The sac was reduced and excised. A polypropylene mesh "
        "was placed and secured with interrupted 2-0 Prolene sutures.\n"
        "\n"
        "Estimated Blood Loss: 25 mL. No complications. The patient tolerated "
        "the procedure well and was transferred to the recovery area in stable "
        "condition.\n"
        "\n"
        "Follow-up: Return visit in two weeks. Call (301) 555-0298 for questions."
    ),
}

# ---- contract_01 ---------------------------------------------------------
DOCUMENTS["contract_01"] = {
    "doc_type": "contract",
    "text": (
        "SERVICE AGREEMENT\n"
        "Agreement No.: CASE-2024-08817\n"
        "Date: August 1, 2024\n"
        "\n"
        "This Service Agreement is entered into by and between Greenfield "
        "Consulting Group, LLC, located at 1200 Commerce Boulevard, Suite 400, "
        "Reston, VA 20190 (\"Provider\"), and Sandra K. Whitmore, individually, "
        "residing at 743 Maple Crest Road, Falls Church, VA 22046 (\"Client\").\n"
        "\n"
        "1. SCOPE OF SERVICES. Provider agrees to deliver market research and "
        "competitive analysis services as described in Exhibit A.\n"
        "\n"
        "2. TERM. This Agreement commences on August 1, 2024 and terminates on "
        "January 31, 2025, unless extended by mutual written consent.\n"
        "\n"
        "3. COMPENSATION. Client shall pay Provider a fixed fee of $45,000, "
        "payable in three equal installments.\n"
        "\n"
        "4. CONTACT. All notices shall be sent to: sandra.whitmore@email.net.\n"
        "\n"
        "IN WITNESS WHEREOF, the parties have executed this Agreement as of "
        "the date first written above.\n"
        "\n"
        "Provider: Thomas J. Greenfield, Managing Partner\n"
        "Client: Sandra K. Whitmore"
    ),
}

# ---- contract_02 ---------------------------------------------------------
DOCUMENTS["contract_02"] = {
    "doc_type": "contract",
    "text": (
        "CONSULTING AGREEMENT\n"
        "Reference: CASE-2024-11503\n"
        "Effective Date: October 15, 2024\n"
        "\n"
        "Between: Apex Data Solutions, Inc., with principal offices at "
        "8900 Technology Parkway, Suite 210, Columbia, MD 21046 (\"Consultant\"), "
        "and Robert M. Vasquez, residing at 334 Birchwood Terrace, Ellicott "
        "City, MD 21043 (\"Client\").\n"
        "\n"
        "1. SERVICES. Consultant shall provide data migration and analytics "
        "platform implementation services for the Client's existing customer "
        "relationship management system.\n"
        "\n"
        "2. DURATION. This Agreement is effective from October 15, 2024 "
        "through April 14, 2025.\n"
        "\n"
        "3. FEES. Client agrees to pay Consultant $72,000 in monthly "
        "installments of $12,000.\n"
        "\n"
        "4. NOTICES. Electronic notices: robert.vasquez@techinbox.com.\n"
        "\n"
        "Signed on behalf of the parties:\n"
        "Consultant: Nina Chen, CEO, Apex Data Solutions, Inc.\n"
        "Client: Robert M. Vasquez"
    ),
}

# ---- resume_01 -----------------------------------------------------------
DOCUMENTS["resume_01"] = {
    "doc_type": "resume",
    "text": (
        "DAVID ALEXANDER BROOKS\n"
        "4521 Riverside Terrace, Baltimore, MD 21230\n"
        "david.brooks@protonmail.com | (443) 555-0187\n"
        "LinkedIn: https://linkedin.com/in/davidabrooks\n"
        "\n"
        "PROFESSIONAL SUMMARY\n"
        "Data engineer with 8 years of experience designing scalable ETL "
        "pipelines and cloud data warehouses. Proficient in Python, SQL, "
        "Apache Spark, and AWS.\n"
        "\n"
        "EXPERIENCE\n"
        "Senior Data Engineer, Meridian Analytics Corp., Baltimore, MD\n"
        "June 2019 - Present\n"
        "- Architected a real-time streaming pipeline processing 2M events/day.\n"
        "- Reduced query latency by 60% through partition optimization.\n"
        "\n"
        "Data Engineer, NovaBridge Technologies, Reston, VA\n"
        "March 2016 - May 2019\n"
        "- Built ETL workflows using Apache Airflow and Spark on EMR.\n"
        "- Migrated 15TB legacy Oracle database to Amazon Redshift.\n"
        "\n"
        "EDUCATION\n"
        "M.S. Computer Science, University of Maryland, Baltimore County, 2016\n"
        "B.S. Mathematics, Towson University, 2014"
    ),
}

# ---- resume_02 -----------------------------------------------------------
DOCUMENTS["resume_02"] = {
    "doc_type": "resume",
    "text": (
        "ELENA MARIA PETROV\n"
        "1038 Harbor View Court, Annapolis, MD 21401\n"
        "elena.petrov@mailhub.org | (240) 555-0342\n"
        "Portfolio: https://elenapetrov.dev/portfolio\n"
        "\n"
        "SUMMARY\n"
        "Full-stack software engineer with 5 years of experience in web "
        "application development. Skilled in React, Node.js, TypeScript, "
        "and PostgreSQL.\n"
        "\n"
        "WORK EXPERIENCE\n"
        "Software Engineer II, Clearpath Digital, Annapolis, MD\n"
        "January 2021 - Present\n"
        "- Led migration of monolithic PHP application to microservices.\n"
        "- Implemented CI/CD pipelines reducing deployment time by 75%.\n"
        "\n"
        "Junior Developer, Bayside Software Co., Columbia, MD\n"
        "August 2019 - December 2020\n"
        "- Developed REST APIs serving 500K daily requests.\n"
        "- Created automated testing suite with 92% code coverage.\n"
        "\n"
        "EDUCATION\n"
        "B.S. Computer Science, University of Maryland, College Park, 2019\n"
        "\n"
        "CERTIFICATIONS\n"
        "AWS Certified Solutions Architect - Associate, March 2022"
    ),
}

# ---- email_01 ------------------------------------------------------------
DOCUMENTS["email_01"] = {
    "doc_type": "email",
    "text": (
        "From: Patricia Huang <patricia.huang@crownpartners.com>\n"
        "To: Michael O'Brien <michael.obrien@bluelinelogistics.com>\n"
        "Date: September 12, 2024\n"
        "Subject: Q3 Partnership Review\n"
        "\n"
        "Dear Michael,\n"
        "\n"
        "Thank you for taking the time to meet with our team at Crown "
        "Partners last Thursday. We were impressed with Blue Line Logistics' "
        "proposal for the mid-Atlantic distribution expansion.\n"
        "\n"
        "As discussed, we would like to schedule a follow-up call for the "
        "week of September 23, 2024. Please coordinate with my assistant, "
        "Karen Liu, at karen.liu@crownpartners.com or (202) 555-0461.\n"
        "\n"
        "I have attached the revised cost projections and the draft SLA for "
        "your review. Please send your comments by September 20, 2024.\n"
        "\n"
        "Best regards,\n"
        "Patricia Huang\n"
        "Director of Strategic Partnerships\n"
        "Crown Partners\n"
        "Direct: (202) 555-0119"
    ),
}

# ---- email_02 ------------------------------------------------------------
DOCUMENTS["email_02"] = {
    "doc_type": "email",
    "text": (
        "From: Derek Lawson <derek.lawson@quantumventures.io>\n"
        "To: Angela Foster <angela.foster@brighthorizon.edu>\n"
        "Date: November 4, 2024\n"
        "Subject: Research Collaboration Proposal\n"
        "\n"
        "Dear Angela,\n"
        "\n"
        "I am writing on behalf of Quantum Ventures to explore a potential "
        "collaboration with Bright Horizon University's artificial "
        "intelligence research lab.\n"
        "\n"
        "Our team has developed a novel approach to federated learning that "
        "we believe could benefit from academic partnership. We would like to "
        "propose a 12-month joint research initiative beginning January 2025.\n"
        "\n"
        "Could we arrange a call next week? You can reach me at "
        "(571) 555-0284 or reply to this email. My colleague, Dr. Yusuf "
        "Osman, will also join; his email is yusuf.osman@quantumventures.io.\n"
        "\n"
        "Looking forward to your response.\n"
        "\n"
        "Sincerely,\n"
        "Derek Lawson\n"
        "VP of Research, Quantum Ventures"
    ),
}

# ---- court_case_01 -------------------------------------------------------
DOCUMENTS["court_case_01"] = {
    "doc_type": "court_case",
    "text": (
        "UNITED STATES DISTRICT COURT\n"
        "DISTRICT OF MARYLAND\n"
        "\n"
        "CIVIL ACTION NO. CASE-2023-04781\n"
        "\n"
        "HELEN MARIE DAWSON,\n"
        "    Plaintiff,\n"
        "v.\n"
        "PINNACLE INSURANCE GROUP, INC.,\n"
        "    Defendant.\n"
        "\n"
        "MEMORANDUM OPINION\n"
        "Filed: March 22, 2024\n"
        "Before: The Honorable Judge Raymond T. Nakamura\n"
        "\n"
        "Plaintiff Helen Marie Dawson initiated this action on July 10, 2023, "
        "alleging that Pinnacle Insurance Group, Inc. wrongfully denied her "
        "long-term disability claim. Plaintiff's Social Security number "
        "(XXX-XX-4829) was referenced in sealed Exhibit 3.\n"
        "\n"
        "The Court finds that the Defendant's denial letter, dated "
        "February 14, 2023, failed to comply with the notice requirements of "
        "29 U.S.C. 1133. Accordingly, the Defendant's motion for summary "
        "judgment is DENIED.\n"
        "\n"
        "SO ORDERED this 22nd day of March, 2024.\n"
        "Raymond T. Nakamura\n"
        "United States District Judge"
    ),
}

# ---- court_case_02 -------------------------------------------------------
DOCUMENTS["court_case_02"] = {
    "doc_type": "court_case",
    "text": (
        "CIRCUIT COURT FOR BALTIMORE COUNTY\n"
        "STATE OF MARYLAND\n"
        "\n"
        "CASE NO. CASE-2024-09156\n"
        "\n"
        "WALTER EUGENE CAMPBELL,\n"
        "    Plaintiff,\n"
        "v.\n"
        "SUMMIT HEALTHCARE SYSTEMS, LLC,\n"
        "    Defendant.\n"
        "\n"
        "ORDER\n"
        "Filed: June 5, 2024\n"
        "Presiding: The Honorable Judge Carolyn A. Westbrook\n"
        "\n"
        "Plaintiff Walter Eugene Campbell filed this medical malpractice action "
        "on January 8, 2024, alleging that Summit Healthcare Systems, LLC "
        "negligently performed a spinal fusion procedure on December 3, 2023. "
        "Plaintiff's Social Security number (XXX-XX-7136) appears in the "
        "sealed medical records filed as Exhibit 7.\n"
        "\n"
        "Upon review of the submissions, the Court grants Plaintiff's motion "
        "to compel production of the operative notes and denies Defendant's "
        "motion to dismiss.\n"
        "\n"
        "IT IS SO ORDERED this 5th day of June, 2024.\n"
        "Carolyn A. Westbrook\n"
        "Circuit Court Judge"
    ),
}


# ---------------------------------------------------------------------------
# PII Span annotations
# ---------------------------------------------------------------------------

def find(text: str, substring: str, occurrence: int = 1) -> int:
    """Find the nth occurrence of substring in text. Raises if not found."""
    start = 0
    for _ in range(occurrence):
        idx = text.find(substring, start)
        if idx == -1:
            raise ValueError(
                f"Occurrence {occurrence} of {substring!r} not found in text"
            )
        start = idx + 1
    return idx


def span(text: str, substring: str, label: str, occurrence: int = 1) -> dict:
    """Create a span dict with verified offsets."""
    start = find(text, substring, occurrence)
    end = start + len(substring)
    actual = text[start:end]
    assert actual == substring, (
        f"Offset mismatch: text[{start}:{end}] = {actual!r}, "
        f"expected {substring!r}"
    )
    return {
        "start": start,
        "end": end,
        "label": label,
        "text": substring,
        "score": 1.0,
        "detector": "gold",
        "validated": False,
    }


# --- medical_01 spans ---
_t = DOCUMENTS["medical_01"]["text"]
DOCUMENTS["medical_01"]["spans"] = [
    span(_t, "Margaret Ellen Sullivan", "PERSON"),
    span(_t, "03/15/1958", "DATE"),
    span(_t, "66", "AGE"),
    span(_t, "MRN-7842193", "MRN"),
    span(_t, "(410) 555-0173", "PHONE"),
    span(_t, "2847 Windmill Lane, Towson, MD 21204", "ADDRESS"),
    span(_t, "Dr. Samuel R. Okonkwo", "PERSON"),
    span(_t, "09/02/2024", "DATE"),
    span(_t, "09/08/2024", "DATE"),
]

# --- medical_02 spans ---
_t = DOCUMENTS["medical_02"]["text"]
DOCUMENTS["medical_02"]["spans"] = [
    span(_t, "James Francis Dellacroce", "PERSON"),
    span(_t, "11/27/1945", "DATE"),
    span(_t, "80", "AGE"),
    span(_t, "MRN-3310567", "MRN"),
    span(_t, "(301) 555-0298", "PHONE", occurrence=1),
    span(_t, "519 Cedar Park Drive, Silver Spring, MD 20910", "ADDRESS"),
    span(_t, "Dr. Priya Anand Mehta", "PERSON"),
    span(_t, "10/14/2024", "DATE"),
    span(_t, "(301) 555-0298", "PHONE", occurrence=2),
]

# --- contract_01 spans ---
_t = DOCUMENTS["contract_01"]["text"]
DOCUMENTS["contract_01"]["spans"] = [
    span(_t, "CASE-2024-08817", "CASE_NUMBER"),
    span(_t, "August 1, 2024", "DATE", occurrence=1),
    span(_t, "Greenfield Consulting Group, LLC", "ORG"),
    span(_t, "1200 Commerce Boulevard, Suite 400, Reston, VA 20190", "ADDRESS"),
    span(_t, "Sandra K. Whitmore", "PERSON", occurrence=1),
    span(_t, "743 Maple Crest Road, Falls Church, VA 22046", "ADDRESS"),
    span(_t, "August 1, 2024", "DATE", occurrence=2),
    span(_t, "January 31, 2025", "DATE"),
    span(_t, "sandra.whitmore@email.net", "EMAIL"),
    span(_t, "Thomas J. Greenfield", "PERSON"),
    span(_t, "Sandra K. Whitmore", "PERSON", occurrence=2),
]

# --- contract_02 spans ---
_t = DOCUMENTS["contract_02"]["text"]
DOCUMENTS["contract_02"]["spans"] = [
    span(_t, "CASE-2024-11503", "CASE_NUMBER"),
    span(_t, "October 15, 2024", "DATE", occurrence=1),
    span(_t, "Apex Data Solutions, Inc.", "ORG", occurrence=1),
    span(_t, "8900 Technology Parkway, Suite 210, Columbia, MD 21046", "ADDRESS"),
    span(_t, "Robert M. Vasquez", "PERSON", occurrence=1),
    span(_t, "334 Birchwood Terrace, Ellicott City, MD 21043", "ADDRESS"),
    span(_t, "October 15, 2024", "DATE", occurrence=2),
    span(_t, "April 14, 2025", "DATE"),
    span(_t, "robert.vasquez@techinbox.com", "EMAIL"),
    span(_t, "Nina Chen", "PERSON"),
    span(_t, "Apex Data Solutions, Inc.", "ORG", occurrence=2),
    span(_t, "Robert M. Vasquez", "PERSON", occurrence=2),
]

# --- resume_01 spans ---
_t = DOCUMENTS["resume_01"]["text"]
DOCUMENTS["resume_01"]["spans"] = [
    span(_t, "DAVID ALEXANDER BROOKS", "PERSON"),
    span(_t, "4521 Riverside Terrace, Baltimore, MD 21230", "ADDRESS"),
    span(_t, "david.brooks@protonmail.com", "EMAIL"),
    span(_t, "(443) 555-0187", "PHONE"),
    span(_t, "https://linkedin.com/in/davidabrooks", "URL"),
    span(_t, "Meridian Analytics Corp.", "ORG"),
    span(_t, "June 2019", "DATE"),
    span(_t, "NovaBridge Technologies", "ORG"),
    span(_t, "March 2016", "DATE", occurrence=1),
    span(_t, "May 2019", "DATE"),
]

# --- resume_02 spans ---
_t = DOCUMENTS["resume_02"]["text"]
DOCUMENTS["resume_02"]["spans"] = [
    span(_t, "ELENA MARIA PETROV", "PERSON"),
    span(_t, "1038 Harbor View Court, Annapolis, MD 21401", "ADDRESS"),
    span(_t, "elena.petrov@mailhub.org", "EMAIL"),
    span(_t, "(240) 555-0342", "PHONE"),
    span(_t, "https://elenapetrov.dev/portfolio", "URL"),
    span(_t, "Clearpath Digital", "ORG"),
    span(_t, "January 2021", "DATE"),
    span(_t, "Bayside Software Co.", "ORG"),
    span(_t, "August 2019", "DATE", occurrence=1),
    span(_t, "December 2020", "DATE"),
    span(_t, "March 2022", "DATE"),
]

# --- email_01 spans ---
_t = DOCUMENTS["email_01"]["text"]
DOCUMENTS["email_01"]["spans"] = [
    span(_t, "Patricia Huang", "PERSON", occurrence=1),
    span(_t, "patricia.huang@crownpartners.com", "EMAIL"),
    span(_t, "Michael O'Brien", "PERSON", occurrence=1),
    span(_t, "michael.obrien@bluelinelogistics.com", "EMAIL"),
    span(_t, "September 12, 2024", "DATE"),
    span(_t, "Crown Partners", "ORG", occurrence=1),
    span(_t, "Blue Line Logistics", "ORG"),
    span(_t, "September 23, 2024", "DATE"),
    span(_t, "Karen Liu", "PERSON"),
    span(_t, "karen.liu@crownpartners.com", "EMAIL"),
    span(_t, "(202) 555-0461", "PHONE"),
    span(_t, "September 20, 2024", "DATE"),
    span(_t, "Patricia Huang", "PERSON", occurrence=2),
    span(_t, "(202) 555-0119", "PHONE"),
]

# --- email_02 spans ---
_t = DOCUMENTS["email_02"]["text"]
DOCUMENTS["email_02"]["spans"] = [
    span(_t, "Derek Lawson", "PERSON", occurrence=1),
    span(_t, "derek.lawson@quantumventures.io", "EMAIL"),
    span(_t, "Angela Foster", "PERSON", occurrence=1),
    span(_t, "angela.foster@brighthorizon.edu", "EMAIL"),
    span(_t, "November 4, 2024", "DATE"),
    span(_t, "Quantum Ventures", "ORG", occurrence=1),
    span(_t, "Bright Horizon University", "ORG"),
    span(_t, "January 2025", "DATE"),
    span(_t, "(571) 555-0284", "PHONE"),
    span(_t, "Dr. Yusuf Osman", "PERSON"),
    span(_t, "yusuf.osman@quantumventures.io", "EMAIL"),
    span(_t, "Derek Lawson", "PERSON", occurrence=2),
    span(_t, "Quantum Ventures", "ORG", occurrence=2),
]

# --- court_case_01 spans ---
_t = DOCUMENTS["court_case_01"]["text"]
DOCUMENTS["court_case_01"]["spans"] = [
    span(_t, "CASE-2023-04781", "CASE_NUMBER"),
    span(_t, "HELEN MARIE DAWSON", "PERSON"),
    span(_t, "PINNACLE INSURANCE GROUP, INC.", "ORG", occurrence=1),
    span(_t, "March 22, 2024", "DATE", occurrence=1),
    span(_t, "Judge Raymond T. Nakamura", "PERSON", occurrence=1),
    span(_t, "Helen Marie Dawson", "PERSON"),
    span(_t, "July 10, 2023", "DATE"),
    span(_t, "Pinnacle Insurance Group, Inc.", "ORG"),
    span(_t, "XXX-XX-4829", "SSN"),
    span(_t, "February 14, 2023", "DATE"),
    span(_t, "Raymond T. Nakamura", "PERSON"),
]

# --- court_case_02 spans ---
_t = DOCUMENTS["court_case_02"]["text"]
DOCUMENTS["court_case_02"]["spans"] = [
    span(_t, "CASE-2024-09156", "CASE_NUMBER"),
    span(_t, "WALTER EUGENE CAMPBELL", "PERSON"),
    span(_t, "SUMMIT HEALTHCARE SYSTEMS, LLC", "ORG", occurrence=1),
    span(_t, "June 5, 2024", "DATE", occurrence=1),
    span(_t, "Judge Carolyn A. Westbrook", "PERSON", occurrence=1),
    span(_t, "Walter Eugene Campbell", "PERSON"),
    span(_t, "January 8, 2024", "DATE"),
    span(_t, "Summit Healthcare Systems, LLC", "ORG"),
    span(_t, "December 3, 2023", "DATE"),
    span(_t, "XXX-XX-7136", "SSN"),
    span(_t, "Carolyn A. Westbrook", "PERSON"),
]


# ---------------------------------------------------------------------------
# QA pairs  (10 per doc = 100 total)
# ---------------------------------------------------------------------------

QA_PAIRS: list[dict] = []

# --- medical_01 QA ---
QA_PAIRS.extend([
    {"qa_id": "med_01_q01", "doc_id": "medical_01", "question": "What was the patient diagnosed with?", "answer": "Acute exacerbation of chronic obstructive pulmonary disease (COPD) with right lower lobe pneumonia.", "answer_type": "non_pii", "notes": ""},
    {"qa_id": "med_01_q02", "doc_id": "medical_01", "question": "What is the patient's name?", "answer": "[PERSON_1]", "answer_type": "pii_dependent", "notes": "Patient name is PII"},
    {"qa_id": "med_01_q03", "doc_id": "medical_01", "question": "How was the pneumonia confirmed?", "answer": "Chest X-ray confirmed right lower lobe consolidation.", "answer_type": "non_pii", "notes": ""},
    {"qa_id": "med_01_q04", "doc_id": "medical_01", "question": "Who is the attending physician?", "answer": "[PERSON_2]", "answer_type": "pii_dependent", "notes": "Doctor name is PII"},
    {"qa_id": "med_01_q05", "doc_id": "medical_01", "question": "What antibiotics were administered during the hospital stay?", "answer": "IV ceftriaxone and azithromycin.", "answer_type": "non_pii", "notes": ""},
    {"qa_id": "med_01_q06", "doc_id": "medical_01", "question": "What is the patient's date of birth?", "answer": "[DATE_1]", "answer_type": "pii_dependent", "notes": "DOB is PII"},
    {"qa_id": "med_01_q07", "doc_id": "medical_01", "question": "How long was the patient hospitalized?", "answer": "Six days.", "answer_type": "non_pii", "notes": "Duration derivable from text without exact dates"},
    {"qa_id": "med_01_q08", "doc_id": "medical_01", "question": "What is the patient's medical record number?", "answer": "[MRN_1]", "answer_type": "pii_dependent", "notes": "MRN is PII"},
    {"qa_id": "med_01_q09", "doc_id": "medical_01", "question": "What supplemental oxygen was provided?", "answer": "Nasal cannula at 2L/min.", "answer_type": "non_pii", "notes": ""},
    {"qa_id": "med_01_q10", "doc_id": "medical_01", "question": "What is the patient's phone number?", "answer": "[PHONE_1]", "answer_type": "pii_dependent", "notes": "Phone is PII"},
])

# --- medical_02 QA ---
QA_PAIRS.extend([
    {"qa_id": "med_02_q01", "doc_id": "medical_02", "question": "What procedure was performed?", "answer": "Open right inguinal hernia repair with mesh (Lichtenstein technique).", "answer_type": "non_pii", "notes": ""},
    {"qa_id": "med_02_q02", "doc_id": "medical_02", "question": "What is the patient's name?", "answer": "[PERSON_1]", "answer_type": "pii_dependent", "notes": "Patient name is PII"},
    {"qa_id": "med_02_q03", "doc_id": "medical_02", "question": "What type of mesh material was used?", "answer": "A polypropylene mesh was placed and secured with interrupted 2-0 Prolene sutures.", "answer_type": "non_pii", "notes": ""},
    {"qa_id": "med_02_q04", "doc_id": "medical_02", "question": "Who performed the surgery?", "answer": "[PERSON_2]", "answer_type": "pii_dependent", "notes": "Surgeon name is PII"},
    {"qa_id": "med_02_q05", "doc_id": "medical_02", "question": "What was the estimated blood loss?", "answer": "25 mL.", "answer_type": "non_pii", "notes": ""},
    {"qa_id": "med_02_q06", "doc_id": "medical_02", "question": "When was the procedure performed?", "answer": "[DATE_3]", "answer_type": "pii_dependent", "notes": "Procedure date is PII"},
    {"qa_id": "med_02_q07", "doc_id": "medical_02", "question": "What was the preoperative diagnosis?", "answer": "Right inguinal hernia, recurrent.", "answer_type": "non_pii", "notes": ""},
    {"qa_id": "med_02_q08", "doc_id": "medical_02", "question": "Where does the patient live?", "answer": "[ADDRESS_1]", "answer_type": "pii_dependent", "notes": "Address is PII"},
    {"qa_id": "med_02_q09", "doc_id": "medical_02", "question": "Were there any complications during the procedure?", "answer": "No complications. The patient tolerated the procedure well.", "answer_type": "non_pii", "notes": ""},
    {"qa_id": "med_02_q10", "doc_id": "medical_02", "question": "What is the patient's medical record number?", "answer": "[MRN_1]", "answer_type": "pii_dependent", "notes": "MRN is PII"},
])

# --- contract_01 QA ---
QA_PAIRS.extend([
    {"qa_id": "con_01_q01", "doc_id": "contract_01", "question": "What services will the Provider deliver?", "answer": "Market research and competitive analysis services as described in Exhibit A.", "answer_type": "non_pii", "notes": ""},
    {"qa_id": "con_01_q02", "doc_id": "contract_01", "question": "Who is the Client named in this agreement?", "answer": "[PERSON_1]", "answer_type": "pii_dependent", "notes": "Client name is PII"},
    {"qa_id": "con_01_q03", "doc_id": "contract_01", "question": "What is the total compensation amount?", "answer": "$45,000, payable in three equal installments.", "answer_type": "non_pii", "notes": ""},
    {"qa_id": "con_01_q04", "doc_id": "contract_01", "question": "What company is the Provider?", "answer": "[ORG_1]", "answer_type": "pii_dependent", "notes": "Company name is PII"},
    {"qa_id": "con_01_q05", "doc_id": "contract_01", "question": "When does this agreement terminate?", "answer": "[DATE_3]", "answer_type": "pii_dependent", "notes": "Date is PII"},
    {"qa_id": "con_01_q06", "doc_id": "contract_01", "question": "How many installments is the payment divided into?", "answer": "Three equal installments.", "answer_type": "non_pii", "notes": ""},
    {"qa_id": "con_01_q07", "doc_id": "contract_01", "question": "What is the agreement number?", "answer": "[CASE_NUMBER_1]", "answer_type": "pii_dependent", "notes": "Case number is PII"},
    {"qa_id": "con_01_q08", "doc_id": "contract_01", "question": "Where should all notices be sent?", "answer": "[EMAIL_1]", "answer_type": "pii_dependent", "notes": "Email is PII"},
    {"qa_id": "con_01_q09", "doc_id": "contract_01", "question": "Can this agreement be extended?", "answer": "Yes, by mutual written consent.", "answer_type": "non_pii", "notes": ""},
    {"qa_id": "con_01_q10", "doc_id": "contract_01", "question": "Who signed as the Managing Partner for the Provider?", "answer": "[PERSON_2]", "answer_type": "pii_dependent", "notes": "Person name is PII"},
])

# --- contract_02 QA ---
QA_PAIRS.extend([
    {"qa_id": "con_02_q01", "doc_id": "contract_02", "question": "What services will the Consultant provide?", "answer": "Data migration and analytics platform implementation services for the Client's existing customer relationship management system.", "answer_type": "non_pii", "notes": ""},
    {"qa_id": "con_02_q02", "doc_id": "contract_02", "question": "Who is the Client?", "answer": "[PERSON_1]", "answer_type": "pii_dependent", "notes": "Client name is PII"},
    {"qa_id": "con_02_q03", "doc_id": "contract_02", "question": "What is the monthly payment amount?", "answer": "$12,000.", "answer_type": "non_pii", "notes": ""},
    {"qa_id": "con_02_q04", "doc_id": "contract_02", "question": "What is the total contract value?", "answer": "$72,000.", "answer_type": "non_pii", "notes": ""},
    {"qa_id": "con_02_q05", "doc_id": "contract_02", "question": "Who is the CEO of the consulting firm?", "answer": "[PERSON_2]", "answer_type": "pii_dependent", "notes": "Person name is PII"},
    {"qa_id": "con_02_q06", "doc_id": "contract_02", "question": "When does the agreement end?", "answer": "[DATE_3]", "answer_type": "pii_dependent", "notes": "Date is PII"},
    {"qa_id": "con_02_q07", "doc_id": "contract_02", "question": "What type of system is being migrated?", "answer": "Customer relationship management system.", "answer_type": "non_pii", "notes": ""},
    {"qa_id": "con_02_q08", "doc_id": "contract_02", "question": "What is the reference number for this agreement?", "answer": "[CASE_NUMBER_1]", "answer_type": "pii_dependent", "notes": "Case number is PII"},
    {"qa_id": "con_02_q09", "doc_id": "contract_02", "question": "Where are the Consultant's offices located?", "answer": "[ADDRESS_1]", "answer_type": "pii_dependent", "notes": "Address is PII"},
    {"qa_id": "con_02_q10", "doc_id": "contract_02", "question": "How is the total fee structured?", "answer": "In monthly installments of $12,000.", "answer_type": "non_pii", "notes": ""},
])

# --- resume_01 QA ---
QA_PAIRS.extend([
    {"qa_id": "res_01_q01", "doc_id": "resume_01", "question": "How many years of experience does the candidate have?", "answer": "8 years.", "answer_type": "non_pii", "notes": ""},
    {"qa_id": "res_01_q02", "doc_id": "resume_01", "question": "What is the candidate's name?", "answer": "[PERSON_1]", "answer_type": "pii_dependent", "notes": "Candidate name is PII"},
    {"qa_id": "res_01_q03", "doc_id": "resume_01", "question": "What programming languages is the candidate proficient in?", "answer": "Python, SQL, Apache Spark, and AWS.", "answer_type": "non_pii", "notes": ""},
    {"qa_id": "res_01_q04", "doc_id": "resume_01", "question": "What is the candidate's email address?", "answer": "[EMAIL_1]", "answer_type": "pii_dependent", "notes": "Email is PII"},
    {"qa_id": "res_01_q05", "doc_id": "resume_01", "question": "By how much did the candidate reduce query latency?", "answer": "60% through partition optimization.", "answer_type": "non_pii", "notes": ""},
    {"qa_id": "res_01_q06", "doc_id": "resume_01", "question": "Where does the candidate currently work?", "answer": "[ORG_1]", "answer_type": "pii_dependent", "notes": "Company name is PII"},
    {"qa_id": "res_01_q07", "doc_id": "resume_01", "question": "What degree did the candidate earn from UMBC?", "answer": "M.S. Computer Science.", "answer_type": "non_pii", "notes": ""},
    {"qa_id": "res_01_q08", "doc_id": "resume_01", "question": "What is the candidate's home address?", "answer": "[ADDRESS_1]", "answer_type": "pii_dependent", "notes": "Address is PII"},
    {"qa_id": "res_01_q09", "doc_id": "resume_01", "question": "How many events per day does the streaming pipeline process?", "answer": "2M events per day.", "answer_type": "non_pii", "notes": ""},
    {"qa_id": "res_01_q10", "doc_id": "resume_01", "question": "What is the candidate's LinkedIn URL?", "answer": "[URL_1]", "answer_type": "pii_dependent", "notes": "URL is PII"},
])

# --- resume_02 QA ---
QA_PAIRS.extend([
    {"qa_id": "res_02_q01", "doc_id": "resume_02", "question": "How many years of experience does the candidate have?", "answer": "5 years.", "answer_type": "non_pii", "notes": ""},
    {"qa_id": "res_02_q02", "doc_id": "resume_02", "question": "What is the candidate's name?", "answer": "[PERSON_1]", "answer_type": "pii_dependent", "notes": "Candidate name is PII"},
    {"qa_id": "res_02_q03", "doc_id": "resume_02", "question": "What technologies is the candidate skilled in?", "answer": "React, Node.js, TypeScript, and PostgreSQL.", "answer_type": "non_pii", "notes": ""},
    {"qa_id": "res_02_q04", "doc_id": "resume_02", "question": "What is the candidate's email address?", "answer": "[EMAIL_1]", "answer_type": "pii_dependent", "notes": "Email is PII"},
    {"qa_id": "res_02_q05", "doc_id": "resume_02", "question": "By how much did CI/CD pipelines reduce deployment time?", "answer": "75%.", "answer_type": "non_pii", "notes": ""},
    {"qa_id": "res_02_q06", "doc_id": "resume_02", "question": "Where is the candidate currently employed?", "answer": "[ORG_1]", "answer_type": "pii_dependent", "notes": "Company name is PII"},
    {"qa_id": "res_02_q07", "doc_id": "resume_02", "question": "What was the code coverage of the automated testing suite?", "answer": "92%.", "answer_type": "non_pii", "notes": ""},
    {"qa_id": "res_02_q08", "doc_id": "resume_02", "question": "What is the candidate's phone number?", "answer": "[PHONE_1]", "answer_type": "pii_dependent", "notes": "Phone is PII"},
    {"qa_id": "res_02_q09", "doc_id": "resume_02", "question": "What certification does the candidate hold?", "answer": "AWS Certified Solutions Architect - Associate.", "answer_type": "non_pii", "notes": ""},
    {"qa_id": "res_02_q10", "doc_id": "resume_02", "question": "What is the candidate's portfolio URL?", "answer": "[URL_1]", "answer_type": "pii_dependent", "notes": "URL is PII"},
])

# --- email_01 QA ---
QA_PAIRS.extend([
    {"qa_id": "eml_01_q01", "doc_id": "email_01", "question": "What was discussed at the meeting last Thursday?", "answer": "Blue Line Logistics' proposal for the mid-Atlantic distribution expansion.", "answer_type": "non_pii", "notes": ""},
    {"qa_id": "eml_01_q02", "doc_id": "email_01", "question": "Who sent this email?", "answer": "[PERSON_1]", "answer_type": "pii_dependent", "notes": "Sender name is PII"},
    {"qa_id": "eml_01_q03", "doc_id": "email_01", "question": "What attachments are included?", "answer": "Revised cost projections and the draft SLA.", "answer_type": "non_pii", "notes": ""},
    {"qa_id": "eml_01_q04", "doc_id": "email_01", "question": "Who is the email addressed to?", "answer": "[PERSON_2]", "answer_type": "pii_dependent", "notes": "Recipient name is PII"},
    {"qa_id": "eml_01_q05", "doc_id": "email_01", "question": "What is the sender's title?", "answer": "Director of Strategic Partnerships.", "answer_type": "non_pii", "notes": ""},
    {"qa_id": "eml_01_q06", "doc_id": "email_01", "question": "Who should be contacted to schedule the follow-up call?", "answer": "[PERSON_3]", "answer_type": "pii_dependent", "notes": "Assistant name is PII"},
    {"qa_id": "eml_01_q07", "doc_id": "email_01", "question": "What is the deadline for sending comments?", "answer": "[DATE_3]", "answer_type": "pii_dependent", "notes": "Date is PII"},
    {"qa_id": "eml_01_q08", "doc_id": "email_01", "question": "What company does the sender work for?", "answer": "[ORG_1]", "answer_type": "pii_dependent", "notes": "Company name is PII"},
    {"qa_id": "eml_01_q09", "doc_id": "email_01", "question": "What is the subject of this email?", "answer": "Q3 Partnership Review.", "answer_type": "non_pii", "notes": ""},
    {"qa_id": "eml_01_q10", "doc_id": "email_01", "question": "What type of expansion is being discussed?", "answer": "Mid-Atlantic distribution expansion.", "answer_type": "non_pii", "notes": ""},
])

# --- email_02 QA ---
QA_PAIRS.extend([
    {"qa_id": "eml_02_q01", "doc_id": "email_02", "question": "What type of collaboration is being proposed?", "answer": "A 12-month joint research initiative on federated learning.", "answer_type": "non_pii", "notes": ""},
    {"qa_id": "eml_02_q02", "doc_id": "email_02", "question": "Who sent this email?", "answer": "[PERSON_1]", "answer_type": "pii_dependent", "notes": "Sender name is PII"},
    {"qa_id": "eml_02_q03", "doc_id": "email_02", "question": "What novel approach has the sender's team developed?", "answer": "A novel approach to federated learning.", "answer_type": "non_pii", "notes": ""},
    {"qa_id": "eml_02_q04", "doc_id": "email_02", "question": "Who is the recipient of this email?", "answer": "[PERSON_2]", "answer_type": "pii_dependent", "notes": "Recipient name is PII"},
    {"qa_id": "eml_02_q05", "doc_id": "email_02", "question": "What is the sender's title?", "answer": "VP of Research.", "answer_type": "non_pii", "notes": ""},
    {"qa_id": "eml_02_q06", "doc_id": "email_02", "question": "Which university is involved in the proposed collaboration?", "answer": "[ORG_2]", "answer_type": "pii_dependent", "notes": "University name is PII"},
    {"qa_id": "eml_02_q07", "doc_id": "email_02", "question": "When would the research initiative begin?", "answer": "[DATE_2]", "answer_type": "pii_dependent", "notes": "Date is PII"},
    {"qa_id": "eml_02_q08", "doc_id": "email_02", "question": "What department at the university is mentioned?", "answer": "Artificial intelligence research lab.", "answer_type": "non_pii", "notes": ""},
    {"qa_id": "eml_02_q09", "doc_id": "email_02", "question": "Who else will join the proposed call?", "answer": "[PERSON_3]", "answer_type": "pii_dependent", "notes": "Colleague name is PII"},
    {"qa_id": "eml_02_q10", "doc_id": "email_02", "question": "What is the subject line of this email?", "answer": "Research Collaboration Proposal.", "answer_type": "non_pii", "notes": ""},
])

# --- court_case_01 QA ---
QA_PAIRS.extend([
    {"qa_id": "crt_01_q01", "doc_id": "court_case_01", "question": "What type of claim was wrongfully denied?", "answer": "Long-term disability claim.", "answer_type": "non_pii", "notes": ""},
    {"qa_id": "crt_01_q02", "doc_id": "court_case_01", "question": "Who is the Plaintiff?", "answer": "[PERSON_1]", "answer_type": "pii_dependent", "notes": "Plaintiff name is PII"},
    {"qa_id": "crt_01_q03", "doc_id": "court_case_01", "question": "What statute did the Defendant violate?", "answer": "29 U.S.C. 1133.", "answer_type": "non_pii", "notes": ""},
    {"qa_id": "crt_01_q04", "doc_id": "court_case_01", "question": "Who is the presiding judge?", "answer": "[PERSON_3]", "answer_type": "pii_dependent", "notes": "Judge name is PII"},
    {"qa_id": "crt_01_q05", "doc_id": "court_case_01", "question": "What was the ruling on the Defendant's motion for summary judgment?", "answer": "DENIED.", "answer_type": "non_pii", "notes": ""},
    {"qa_id": "crt_01_q06", "doc_id": "court_case_01", "question": "What is the civil action number?", "answer": "[CASE_NUMBER_1]", "answer_type": "pii_dependent", "notes": "Case number is PII"},
    {"qa_id": "crt_01_q07", "doc_id": "court_case_01", "question": "What court is hearing this case?", "answer": "United States District Court, District of Maryland.", "answer_type": "non_pii", "notes": ""},
    {"qa_id": "crt_01_q08", "doc_id": "court_case_01", "question": "Who is the Defendant?", "answer": "[ORG_1]", "answer_type": "pii_dependent", "notes": "Company name is PII"},
    {"qa_id": "crt_01_q09", "doc_id": "court_case_01", "question": "What did the Defendant's denial letter fail to do?", "answer": "Failed to comply with the notice requirements of 29 U.S.C. 1133.", "answer_type": "non_pii", "notes": ""},
    {"qa_id": "crt_01_q10", "doc_id": "court_case_01", "question": "In which exhibit was the Plaintiff's Social Security number referenced?", "answer": "Sealed Exhibit 3.", "answer_type": "non_pii", "notes": "Exhibit number is not PII"},
])

# --- court_case_02 QA ---
QA_PAIRS.extend([
    {"qa_id": "crt_02_q01", "doc_id": "court_case_02", "question": "What type of legal action is this?", "answer": "Medical malpractice action.", "answer_type": "non_pii", "notes": ""},
    {"qa_id": "crt_02_q02", "doc_id": "court_case_02", "question": "Who is the Plaintiff?", "answer": "[PERSON_1]", "answer_type": "pii_dependent", "notes": "Plaintiff name is PII"},
    {"qa_id": "crt_02_q03", "doc_id": "court_case_02", "question": "What procedure is alleged to have been negligently performed?", "answer": "A spinal fusion procedure.", "answer_type": "non_pii", "notes": ""},
    {"qa_id": "crt_02_q04", "doc_id": "court_case_02", "question": "Who is the presiding judge?", "answer": "[PERSON_3]", "answer_type": "pii_dependent", "notes": "Judge name is PII"},
    {"qa_id": "crt_02_q05", "doc_id": "court_case_02", "question": "What ruling did the Court make on the motion to compel?", "answer": "Granted.", "answer_type": "non_pii", "notes": ""},
    {"qa_id": "crt_02_q06", "doc_id": "court_case_02", "question": "What is the case number?", "answer": "[CASE_NUMBER_1]", "answer_type": "pii_dependent", "notes": "Case number is PII"},
    {"qa_id": "crt_02_q07", "doc_id": "court_case_02", "question": "What court is hearing this case?", "answer": "Circuit Court for Baltimore County, State of Maryland.", "answer_type": "non_pii", "notes": ""},
    {"qa_id": "crt_02_q08", "doc_id": "court_case_02", "question": "Who is the Defendant?", "answer": "[ORG_1]", "answer_type": "pii_dependent", "notes": "Company name is PII"},
    {"qa_id": "crt_02_q09", "doc_id": "court_case_02", "question": "What was the ruling on the motion to dismiss?", "answer": "Denied.", "answer_type": "non_pii", "notes": ""},
    {"qa_id": "crt_02_q10", "doc_id": "court_case_02", "question": "What documents does the Plaintiff want produced?", "answer": "The operative notes.", "answer_type": "non_pii", "notes": ""},
])


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_all() -> bool:
    """Verify every span offset matches the document text. Return True if OK."""
    ok = True
    for doc_id, doc in DOCUMENTS.items():
        text = doc["text"]
        for i, s in enumerate(doc["spans"]):
            actual = text[s["start"]:s["end"]]
            if actual != s["text"]:
                print(
                    f"MISMATCH {doc_id} span[{i}]: "
                    f"expected {s['text']!r}, got {actual!r} "
                    f"at [{s['start']}:{s['end']}]",
                    file=sys.stderr,
                )
                ok = False
    return ok


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_samples() -> None:
    """Write .txt files to data/samples/."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for doc_id, doc in DOCUMENTS.items():
        path = SAMPLES_DIR / f"{doc_id}.txt"
        path.write_text(doc["text"], encoding="utf-8")
        print(f"  wrote {path} ({len(doc['text'])} chars)")


def write_gold_jsonl() -> None:
    """Write data/gold/handlabel.jsonl."""
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    path = GOLD_DIR / "handlabel.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for doc_id, doc in DOCUMENTS.items():
            record = {
                "doc_id": doc_id,
                "text": doc["text"],
                "spans": doc["spans"],
                "doc_type": doc["doc_type"],
                "language": "en",
                "source": "handlabel",
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  wrote {path} ({len(DOCUMENTS)} docs)")


def write_qa_jsonl() -> None:
    """Write data/gold/qa_template.jsonl."""
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    path = GOLD_DIR / "qa_template.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for qa in QA_PAIRS:
            fh.write(json.dumps(qa, ensure_ascii=False) + "\n")
    print(f"  wrote {path} ({len(QA_PAIRS)} QA pairs)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("Verifying span offsets...")
    if not verify_all():
        print("FAILED: span offset mismatches detected.", file=sys.stderr)
        return 1
    print("All spans verified OK.\n")

    # Statistics
    total_spans = sum(len(d["spans"]) for d in DOCUMENTS.values())
    pii_dep = sum(1 for qa in QA_PAIRS if qa["answer_type"] == "pii_dependent")
    non_pii = sum(1 for qa in QA_PAIRS if qa["answer_type"] == "non_pii")
    print(f"Documents: {len(DOCUMENTS)}")
    print(f"Total PII spans: {total_spans}")
    print(f"QA pairs: {len(QA_PAIRS)} ({pii_dep} pii_dependent, {non_pii} non_pii)\n")

    print("Writing sample documents...")
    write_samples()
    print()

    print("Writing gold annotations...")
    write_gold_jsonl()
    print()

    print("Writing QA template...")
    write_qa_jsonl()
    print()

    # Final verification: re-read and check
    print("Re-reading and verifying handlabel.jsonl...")
    handlabel_path = GOLD_DIR / "handlabel.jsonl"
    errors = 0
    with handlabel_path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            record = json.loads(line)
            text = record["text"]
            for i, s in enumerate(record["spans"]):
                actual = text[s["start"]:s["end"]]
                if actual != s["text"]:
                    print(
                        f"  MISMATCH line {line_no} span[{i}]: "
                        f"{s['text']!r} != {actual!r}",
                        file=sys.stderr,
                    )
                    errors += 1
    if errors:
        print(f"FAILED: {errors} mismatches in written file.", file=sys.stderr)
        return 1

    print("All written data verified. Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
