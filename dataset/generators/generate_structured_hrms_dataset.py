import json
import random
from pathlib import Path
from datetime import datetime, timedelta

EMPLOYEES_FILE = Path("dataset/raw/employees.json")
TASKS_FILE = Path("dataset/raw/task_templates.json")
OUTPUT_FILE = Path("dataset/processed/intelligent_manager_4000.jsonl")

TODAY = datetime(2026, 2, 6)

# ----------------------------
# PHRASE POOLS (HIGH ENTROPY)
# ----------------------------

OVERDUE_PHRASES = [
    "Several commitments have exceeded their deadlines.",
    "There are delayed deliverables requiring attention.",
    "Overdue tasks indicate execution slippage.",
    "Missed timelines detected across assigned work.",
    "Deadlines have been breached for active tasks."
]

UPCOMING_PHRASES = [
    "Critical deadlines are approaching within the next 48 hours.",
    "Upcoming due dates require monitoring.",
    "Short-term delivery pressure is building.",
    "Time-sensitive tasks are nearing completion windows."
]

STABLE_PHRASES = [
    "No immediate risk indicators detected.",
    "Current workload appears manageable.",
    "Task execution remains within expected timelines.",
    "Operational stability maintained."
]

ESCALATION_PHRASES = [
    "Escalation is recommended if delays persist.",
    "Manager intervention may be required.",
    "Workload rebalancing should be considered.",
    "Immediate corrective action is advisable."
]

ACTION_POOL = [
    "Prioritize overdue items and update stakeholders.",
    "Reallocate workload to maintain delivery timelines.",
    "Conduct progress review with assigned employee.",
    "Escalate unresolved delays to reporting manager.",
    "Adjust project timeline based on execution status."
]


# ----------------------------
# LOAD DATA
# ----------------------------

def load_data():
    with open(EMPLOYEES_FILE, "r") as f:
        employees = json.load(f)

    with open(TASKS_FILE, "r") as f:
        task_templates = json.load(f)

    return employees, task_templates


# ----------------------------
# TASK GENERATION
# ----------------------------

def random_due_date():
    delta = random.randint(-20, 15)
    return (TODAY + timedelta(days=delta)).strftime("%Y-%m-%d")


def generate_tasks(task_templates):
    tasks = []
    category = random.choice(list(task_templates.keys()))
    templates = task_templates[category]

    for _ in range(random.randint(2, 6)):
        tasks.append({
            "title": random.choice(templates),
            "status": random.choice(["completed", "in-progress", "pending"]),
            "dueDate": random_due_date()
        })

    return tasks


# ----------------------------
# INTELLIGENT RISK BUILDER
# ----------------------------

def build_manager_output(tasks):

    overdue = 0
    upcoming = 0
    completed = 0

    for t in tasks:
        due = datetime.strptime(t["dueDate"], "%Y-%m-%d")

        if t["status"] == "completed":
            completed += 1
        elif due < TODAY:
            overdue += 1
        elif (due - TODAY).days <= 2:
            upcoming += 1

    # ---- Risk Level ----
    if overdue >= 3:
        risk_level = "high"
    elif overdue >= 1 or upcoming >= 2:
        risk_level = "medium"
    else:
        risk_level = "low"

    # ---- Summary ----
    summary_parts = []

    if overdue:
        summary_parts.append(random.choice(OVERDUE_PHRASES))

    if upcoming:
        summary_parts.append(random.choice(UPCOMING_PHRASES))

    if not overdue and not upcoming:
        summary_parts.append(random.choice(STABLE_PHRASES))

    summary = " ".join(summary_parts)

    # ---- Manager Note ----
    manager_note = (
        f"{completed} tasks completed. "
        f"{overdue} overdue. "
        f"{upcoming} approaching deadlines."
    )

    # ---- Recommended Action ----
    if risk_level == "high":
        action = random.choice(ESCALATION_PHRASES)
    else:
        action = random.choice(ACTION_POOL)

    return {
        "risk_level": risk_level,
        "summary": summary,
        "manager_note": manager_note,
        "recommended_action": action
    }


# ----------------------------
# RECORD BUILDER
# ----------------------------

def build_record(system, user_input, assistant_output):
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": json.dumps(assistant_output)}
        ]
    }


# ----------------------------
# MAIN
# ----------------------------

def main():
    employees, task_templates = load_data()

    system_prompt = (
        "You are an intelligent HRMS Manager AI. "
        "You analyze structured HRMS state and produce structured risk assessment."
    )

    with open(OUTPUT_FILE, "w") as out:
        for _ in range(4000):
            employee = random.choice(employees)
            tasks = generate_tasks(task_templates)

            user_input = (
                "Analyze HRMS state and provide structured risk evaluation.\n\n"
                f"INPUT:\n{json.dumps({'employee': employee, 'tasks': tasks, 'today': TODAY.strftime('%Y-%m-%d')})}"
            )

            assistant_output = build_manager_output(tasks)

            record = build_record(system_prompt, user_input, assistant_output)
            out.write(json.dumps(record) + "\n")

    print("Intelligent dataset generated:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
