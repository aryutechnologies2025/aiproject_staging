# hrms_services.py
import httpx
from datetime import datetime, timezone
from typing import Dict, List

EMPLOYEE_API = "https://hrms.aryutechnologies.com/api/employees/all-active-employees"
TASK_API = "https://hrms.aryutechnologies.com/api/task/all-tasklist"

TIMEOUT = 15.0

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
    OVERDUE_DAYS = 5  # 🔒 FIXED RULE

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

        # ✅ ONLY createdAt (or dueDate if exists)
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

        # ✅ FINAL RULE
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


