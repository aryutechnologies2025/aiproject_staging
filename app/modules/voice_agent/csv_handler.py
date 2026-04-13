import csv
import io
import os
from datetime import datetime
from typing import List, Tuple
from fastapi import UploadFile

from app.modules.voice_agent import config
from app.modules.voice_agent import database as db
from app.modules.voice_agent.schemas import CSVLeadRow

REQUIRED_COLUMNS = {"name", "phone"}
OPTIONAL_COLUMNS = {"qualification", "experience_years", "language_preference", "company_id"}


async def process_csv_upload(file: UploadFile, company_id: str) -> Tuple[int, int, List[str]]:
    content = await file.read()

    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = content.decode("latin-1")

    reader = csv.DictReader(io.StringIO(text))
    headers = {h.strip().lower() for h in (reader.fieldnames or [])}

    missing = REQUIRED_COLUMNS - headers
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    filename = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    os.makedirs(config.UPLOADS_DIR, exist_ok=True)
    save_path = os.path.join(config.UPLOADS_DIR, filename)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text)

    leads_to_create = []
    errors = []
    row_num = 1

    for row in reader:
        row_num += 1
        clean = {k.strip().lower(): v.strip() for k, v in row.items()}

        row_company_id = clean.get("company_id", "").strip() or company_id

        try:
            validated = CSVLeadRow(
                name=clean.get("name", ""),
                phone=clean.get("phone", ""),
                qualification=clean.get("qualification"),
                experience_years=clean.get("experience_years"),
                language_preference=clean.get("language_preference", "ta"),
            )
            if not validated.name or not validated.phone:
                errors.append(f"Row {row_num}: name or phone is empty")
                continue

            leads_to_create.append({
                "name": validated.name,
                "phone": validated.phone,
                "company_id": row_company_id,
                "qualification": validated.qualification,
                "experience_years": int(validated.experience_years) if validated.experience_years else None,
                "language_preference": validated.language_preference or "ta",
                "source_file": filename,
                "status": "pending",
            })
        except Exception as e:
            errors.append(f"Row {row_num}: {str(e)}")

    created = await db.bulk_create_leads(leads_to_create)
    skipped = len(leads_to_create) - created

    return created, skipped, errors