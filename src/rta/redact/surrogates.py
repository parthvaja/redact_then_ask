"""Surrogate generation for ``fake`` redaction mode (Phase 8).

Uses ``Faker`` seeded by cluster id for consistent surrogates.
"""

from __future__ import annotations

from faker import Faker

from rta.config import get_config

_LABEL_GENERATORS: dict[str, str] = {
    "PERSON": "name",
    "ORG": "company",
    "LOCATION": "city",
    "ADDRESS": "street_address",
    "EMAIL": "email",
    "PHONE": "phone_number",
    "SSN": "ssn",
    "CREDIT_CARD": "credit_card_number",
    "IP_ADDRESS": "ipv4",
    "URL": "url",
    "DATE": "date",
    "AGE": "_age",
    "ID_NUMBER": "_id_number",
    "MRN": "_mrn",
    "CASE_NUMBER": "_case_number",
    "IBAN": "_iban",
}


def generate_surrogate(label: str, cluster_id: int) -> str:
    """Generate a realistic fake value for *label*, seeded by *cluster_id*."""
    cfg = get_config()
    seed = cfg.seed + cluster_id
    fake = Faker()
    Faker.seed(seed)

    method = _LABEL_GENERATORS.get(label)
    if method is None:
        return f"[{label}_{cluster_id}]"

    if method.startswith("_"):
        return _custom_generate(fake, method, cluster_id)

    return getattr(fake, method)()


def _custom_generate(fake: Faker, method: str, cluster_id: int) -> str:
    if method == "_age":
        return str(fake.random_int(18, 90))
    if method == "_id_number":
        return f"ID-{fake.random_int(100000, 999999)}"
    if method == "_mrn":
        return f"MRN-{fake.random_int(100000, 999999)}"
    if method == "_case_number":
        return f"{fake.random_int(20, 26)}-CV-{fake.random_int(1000, 9999)}"
    if method == "_iban":
        return f"GB{fake.random_int(10, 99)}MOCK{fake.random_int(1000000, 9999999)}"
    return f"[{method}_{cluster_id}]"
