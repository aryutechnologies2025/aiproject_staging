# hrms_ai/monitoring/workload_analyzer.py

def analyze_workload(data: dict):
    active = data.get("activeTasks", 0)
    average = data.get("averageTaskLoad", 0)

    if active > average * 1.5:
        return {
            "risk": "high",
            "recommendation": "Rebalance workload"
        }

    return {
        "risk": "normal",
        "recommendation": "No action needed"
    }
