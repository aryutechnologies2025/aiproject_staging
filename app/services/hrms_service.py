# hrms_services.py
import httpx
from datetime import datetime, timezone
from typing import Dict, List
from bs4 import BeautifulSoup
from datetime import date, timedelta
from app.services.llm_client import call_llm
from app.services.prompt_service import get_prompt
import httpx
import re
import json
from datetime import datetime, timedelta

EMPLOYEE_API = "https://hrms.aryuprojects.com/api/employees/all-active-employees"
TASK_API = "https://hrms.aryuprojects.com/api/task/all-tasklist"
PROJECT_API = "https://hrms.aryuprojects.com/api/project/view-projects"
HRMS_LOGIN_URL = "https://hrms.aryuprojects.com/api/auth/login/admin"
TASK_CREATE_API = "https://hrms.aryuprojects.com/api/task/create-task"

TIMEOUT = 15.0


_token_cache = {
    "token": None,
    "expires_at": None
}

async def get_hrms_token() -> str:
    # reuse token if still valid
    if _token_cache["token"] and _token_cache["expires_at"]:
        if datetime.utcnow() < _token_cache["expires_at"]:
            return _token_cache["token"]

    payload = {
        "email": "venu@aryutechnologies.com",
        "password": "venu638"
    }

    async with httpx.AsyncClient(timeout=10) as client:
        res = await client.post(HRMS_LOGIN_URL, json=payload)
        res.raise_for_status()

        data = res.json()

        token = data["accessToken"]
        expires_in = data.get("expiresIn", 3600)

        _token_cache["token"] = token
        _token_cache["expires_at"] = datetime.utcnow() + timedelta(seconds=expires_in - 60)

        return token

def clean_project_description(html: str) -> str:
    if not html:
        return "No project description provided."

    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator=" ").strip()

def build_project_context(project: dict) -> str:
    description = clean_project_description(project["description"])

    return f"""
Project Name: {project['name']}

Project Description:
{description}

Project Timeline:
Start Date: {project['startDate']}
End Date: {project['endDate']}
""".strip()

async def fetch_employees() -> Dict[str, dict]:
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        res = await client.get(EMPLOYEE_API)

        res.raise_for_status()

    employees = {}
    for emp in res.json()["data"]:
        employees[emp["_id"]] = {
            "employee_id": emp["_id"],
            "employeeName": emp["employeeName"],
            "email": emp.get("email"),
            "phone": emp.get("phoneNumber"),
            "employeeCode": emp.get("employeeId"),
            "department": emp.get("role", {}).get("department", {}).get("name"),
            "role": emp.get("role", {}).get("name"),
        }

    return employees

async def fetch_tasks() -> List[dict]:
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        res = await client.get(TASK_API)
        res.raise_for_status()

    return res.json()["data"]

def calculate_overdue_tasks(
    tasks: List[dict],
    employees: Dict[str, dict],
    project_name: str | None = None
):
    today = datetime.now(timezone.utc)
    OVERDUE_DAYS = 5  # FIXED RULE

    result = {}

    for task in tasks:
        # only unfinished tasks
        if task.get("status") not in ["todo", "in-progress"]:
            continue

        # project filter (case-insensitive)
        project = task.get("projectId", {}).get("name", "").strip().lower()
        if project_name and project != project_name.strip().lower():
            continue

        # assignment check
        assigned = task.get("assignedTo")
        if not assigned:
            continue

        emp_id = assigned.get("_id")
        if emp_id not in employees:
            continue

        # ONLY createdAt (or dueDate if exists)
        raw_date = task.get("dueDate") or task.get("createdAt")
        if not raw_date:
            continue

        created_date = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
        pending_days = (today - created_date).days

        print(
            f"TASK {task.get('taskId')} | "
            f"createdAt={task.get('createdAt')} | "
            f"pending_days={pending_days}"
        )

        # FINAL RULE
        if pending_days <= OVERDUE_DAYS:
            continue

        result.setdefault(emp_id, {
            **employees[emp_id],
            "tasks": []
        })

        result[emp_id]["tasks"].append({
            "taskId": task["taskId"],
            "title": task["title"],
            "project": task.get("projectId", {}).get("name"),
            "pending_days": pending_days,
            "priority": task.get("priority"),
            "status": task.get("status")
        })

    return list(result.values())

def normalize_task_from_ai(
    ai_response: dict,
    project: dict,
    assigned_to: str,
    created_by_id: str
) -> dict:
    """
    Converts AI response into a valid HRMS task payload.
    Description is ALWAYS a plain string.
    """

    # -------- Title --------
    title = ai_response.get("title")
    if not isinstance(title, str) or not title.strip():
        raise Exception("AI response missing valid title")

    title = title.strip()

    # -------- FORCE description to STRING --------
    raw_description = ai_response.get("description")

    if isinstance(raw_description, str):
        description = raw_description.strip()

    elif isinstance(raw_description, (list, tuple, set)):
        description = "\n".join(
            str(item).strip() for item in raw_description if str(item).strip()
        )

    elif isinstance(raw_description, dict):
        description = "\n".join(
            str(value).strip() for value in raw_description.values() if str(value).strip()
        )

    else:
        description = ""

    if not description:
        raise Exception("AI response missing valid description")

    # -------- Priority --------
    priority = ai_response.get("priority", "medium")
    if isinstance(priority, str):
        priority = priority.lower()

    if priority not in {"low", "medium", "high"}:
        priority = "medium"

    # -------- Task Type --------
    task_type = ai_response.get("taskType", "newRequirement")

    # -------- Due Date --------
    due_date = ai_response.get("dueDate")

    if isinstance(due_date, str) and "T" in due_date:
        due_date = due_date.split("T")[0]

    if not due_date:
        due_date = (datetime.utcnow().date() + timedelta(days=3)).isoformat()

    # -------- Project End Date Safety --------
    project_end = project.get("endDate")
    if project_end:
        try:
            project_end_date = datetime.fromisoformat(
                project_end.replace("Z", "")
            ).date()

            if datetime.fromisoformat(due_date).date() > project_end_date:
                due_date = project_end_date.isoformat()
        except Exception:
            pass

    # -------- FINAL HRMS PAYLOAD --------
    return {
        "startDate": datetime.utcnow().date().isoformat(),
        "dueDate": due_date,
        "projectName": project["name"],
        "description": description,   # ALWAYS STRING
        "status": "todo",
        "title": title,
        "assignedTo": assigned_to,
        "createdById": created_by_id,
        "priority": priority,
        "taskType": task_type,
        "projectId": project["projectId"],
        "projectManagerId": project["projectManager"],
    }


# async def fetch_projects(token: str) -> dict:
#     async with httpx.AsyncClient(timeout=TIMEOUT) as client:
#         res = await client.get(
#             PROJECT_API,
#             headers={"Authorization": f"Bearer {token}"}
#         )
#         res.raise_for_status()

#         projects = {}
#         for p in res.json()["data"]:
#             projects[p["_id"]] = {
#                 "projectId": p["_id"],
#                 "name": p["name"],
#                 "description": p.get("projectDescription", ""),
#                 "projectManager": p.get("projectManager"),
#                 "startDate": p.get("startDate"),
#                 "endDate": p.get("endDate"),
#             }

#         return projects

async def fetch_projects() -> dict:
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        res = await client.get(PROJECT_API)
        res.raise_for_status()

        projects = {}
        for p in res.json()["data"]:
            projects[p["_id"]] = {
                "projectId": p["_id"],
                "name": p["name"],
                "description": p.get("projectDescription", ""),
                "projectManager": p.get("projectManager"),
                "startDate": p.get("startDate"),
                "endDate": p.get("endDate"),
            }
        # print(f"Fetched {projects} projects from HRMS")
        return projects

async def create_task(payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=10) as client:
        res = await client.post(
            TASK_CREATE_API,
            json=payload
        )
        res.raise_for_status()
        return res.json()

def extract_json_from_text(text: str) -> dict:
    """
    Extract first valid JSON object from AI response text.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in AI response")

    json_str = match.group(0)

    # Remove JS-style comments if any
    json_str = re.sub(r"//.*", "", json_str)

    return json.loads(json_str)

async def create_task_via_ai(
    db,
    user_prompt: str,
    project_name: str,
    assigned_to: str,
    created_by_id: str
):
    # 1️⃣ Load master HRMS prompt
    system_prompt = await get_prompt(db, agent_name="hrms_management")
    if not system_prompt:
        raise Exception("HRMS master prompt not found")

    # 2️⃣ Fetch projects (NO AUTH)
    projects = await fetch_projects()

    # 3️⃣ Match project
    project = next(
        (p for p in projects.values()
         if p["name"].lower() == project_name.lower()),
        None
    )

    if not project:
        raise Exception(f"Project '{project_name}' not found")

    # 4️⃣ Build project context
    project_context = build_project_context(project)

    # 5️⃣ Build final prompt
    final_prompt = f"""
{system_prompt}

CURRENT MODULE: Task Management

PROJECT CONTEXT:
{project_context}

USER REQUEST:
{user_prompt}

OUTPUT FORMAT:
{{
  "title": "",
  "description": "",
  "priority": "",
  "dueDate": "",
  "taskType": "newRequirement"
}}
""".strip()

    # 6️⃣ Call LLM
    raw_response = await call_llm(
        user_message=final_prompt,
        agent_name="hrms_management",
        db=db
    )
    print(f"AI Response: {raw_response}")

    try:
        ai_response = extract_json_from_text(raw_response)
    except Exception as e:
        raise Exception(f"AI response JSON parsing failed-{str(e)}")

    # 7️⃣ Build task payload
    today = date.today().isoformat()

    task_payload = normalize_task_from_ai(
        ai_response=ai_response,
        project=project,
        assigned_to=assigned_to,
        created_by_id=created_by_id
    )

    # 8️⃣ Create task (NO AUTH)
    return await create_task(task_payload)


def enforce_bullets(text: str) -> str:
    text = text.strip()

    if not text:
        return ""

    # Already bullet formatted
    if text.startswith("-"):
        return text

    # Convert sentence → bullets (8B-safe)
    parts = [
        p.strip()
        for p in text.replace(".", ".\n").split("\n")
        if p.strip()
    ]

    bullets = [f"- {p.rstrip('.')}" for p in parts[:4]]
    return "\n".join(bullets)


async def describe_task_from_title(
    db,
    title: str,
    project_name: str
) -> str:
    # Load master prompt
    system_prompt = await get_prompt(db, agent_name="hrms_management")
    if not system_prompt:
        raise Exception("HRMS master prompt not found")

    # Fetch projects
    projects = await fetch_projects()

    project = next(
        (p for p in projects.values()
         if p["name"].lower() == project_name.lower()),
        None
    )
    if not project:
        raise Exception(f"Project '{project_name}' not found")

    # Build context
    project_context = build_project_context(project)

    # Prompt
    final_prompt = f"""
{system_prompt}

MODE: DESCRIBE_TASK_TITLE
CURRENT MODULE: Task Management

PROJECT CONTEXT:
{project_context}

TASK TITLE:
{title}

OUTPUT FORMAT:
{{
  "description": ""
}}
""".strip()

    # Call LLM
    raw_response = await call_llm(
        user_message=final_prompt,
        agent_name="hrms_management",
        db=db
    )

    if not raw_response or not raw_response.strip():
        raise Exception("Empty AI response")

    raw_response = raw_response.strip()

    # SAFE extraction (NO hard failure)
    description = None

    try:
        parsed = extract_json_from_text(raw_response)
        description = parsed.get("description")
    except Exception:
        # Ignore JSON errors completely
        description = raw_response

    # Final validation
    if not isinstance(description, str) or not description.strip():
        raise Exception("Invalid description generated")

    # Enforce bullets
    description = enforce_bullets(description)

    return description

