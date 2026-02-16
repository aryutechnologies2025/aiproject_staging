# api/v1/hrms.py

from fastapi import APIRouter, Query, Depends, Form, Body
from typing import Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from hrms_ai.engine.manager_ai import HRMSManagerAI
from ...services.hrms_service import (
    fetch_employees,
    fetch_tasks,
    calculate_overdue_tasks,
    create_task_via_ai,
    describe_task_from_title,
    generate_project_requirements_from_text,
)
from app.core.database import get_db

router = APIRouter(prefix="/api/ai/hrms", tags=["HRMS AI"])

@router.get("/overdue-employees")
async def overdue_employees(
    project: str | None = None,
    db: AsyncSession = Depends(get_db)
):
    employees = await fetch_employees()
    tasks = await fetch_tasks()
    print(f'employee: {len(employees)}')
    print(f'task: {len(tasks)}')

    overdue = calculate_overdue_tasks(
        tasks=tasks,
        employees=employees,
        project_name=project
    )
    print(f'overdue: {len(overdue)}')
    return {
        "count": len(overdue),
        "employees": overdue
    }


@router.post("/send-alerts")
async def send_alerts(
    employees: list,
    admin_id: str,
    db: AsyncSession = Depends(get_db)
):
    from ...models.hrms_alert import AlertRequest, AlertMessage

    alert = AlertRequest(
        admin_id=admin_id,
        total_employees=len(employees),
        status="sent"
    )
    db.add(alert)
    await db.commit()
    await db.refresh(alert)

    for emp in employees:
        message = (
            "Pending Task Reminder\n\n"
            "You have tasks pending for more than 3 days.\n"
            "Please reply with reason or expected completion date."
        )

        db.add(AlertMessage(
            alert_request_id=alert.id,
            employee_id=emp["employee_id"],
            employee_name=emp["employeeName"],
            message=message
        ))

        # Trigger WhatsApp / Email / App notification here

    await db.commit()
    return {"success": True, "alert_id": alert.id}

@router.post("/employee-response")
async def employee_response(
    employee_id: str,
    employee_name: str,
    response: str,
    db: AsyncSession = Depends(get_db)
):
    from ...models.hrms_alert import AlertResponse

    db.add(AlertResponse(
        employee_id=employee_id,
        employee_name=employee_name,
        response=response
    ))
    await db.commit()

    return {"success": True}

@router.get("/admin/alert-responses")
async def admin_responses(db: AsyncSession = Depends(get_db)):
    rows = await db.execute(
        "SELECT employee_name, response, received_at FROM alert_responses ORDER BY received_at DESC"
    )
    return rows.fetchall()

@router.post("/task")
async def ai_create_task(payload: dict, db: AsyncSession = Depends(get_db)):
    return await create_task_via_ai(
        db=db,
        user_prompt=payload["prompt"],
        project_name=payload["projectName"],
        assigned_to=payload["assignedTo"],
        created_by_id=payload["createdById"]
    )

@router.post("/tasks-description")
async def ai_create_task_description(payload: dict, db: AsyncSession = Depends(get_db)):
    description = await describe_task_from_title(
        db=db,
        title=payload["title"],
        project_name=payload["projectName"]
    )
    return {
        "description": description
    }


@router.post("/project-requirements")
async def ai_generate_project_requirements(
    payload: Optional[Dict[str, Any]] = Body(None),
    projectName: Optional[str] = Form(None),
    rawText: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    """
    Accepts BOTH:
    - application/json
    - multipart/form-data
    """

    # ---- Normalize input ----
    if payload:
        project_name = payload.get("projectName")
        raw_text = payload.get("rawText")
    else:
        project_name = projectName
        raw_text = rawText

    if not project_name or not raw_text:
        raise ValueError("projectName and rawText are required")

    requirements = await generate_project_requirements_from_text(
        db=db,
        project_name=project_name,
        raw_text=raw_text
    )

    return {
        "success": True,
        "data": {
            "projectName": project_name,
            "projectRequirements": requirements
        }
    }

@router.post("/test-task")
async def test_task_ai(payload: dict, db: AsyncSession = Depends(get_db)):
    ai = HRMSManagerAI()
    decision = await ai.process_task(payload, db=db)

    return {
        "success": True,
        "decision": decision.dict()
    }

