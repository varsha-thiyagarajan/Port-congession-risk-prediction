"""Step 6 - the decision framework: what each client does at Low / Medium / High congestion."""
import pandas as pd

CLIENTS = ["Port authority", "Shipping company", "Logistics planner", "Marine operations"]

ACTIONS = {
    "Low": {
        "Port authority": "Normal berth schedule. Use the quiet window for maintenance and dredging checks.",
        "Shipping company": "Sail at planned speed. Confirm berth window as usual.",
        "Logistics planner": "Plan trucks and rail on the vessel's published ETA.",
        "Marine operations": "Standard pilot and tug roster.",
    },
    "Medium": {
        "Port authority": "Re-sequence berth slots so ships already waiting get priority. Publish the day-ahead forecast.",
        "Shipping company": "Ask for a confirmed berth window before final approach. Prepare to slow-steam to save fuel.",
        "Logistics planner": "Add a buffer of several hours to container pickup times and book extra chassis.",
        "Marine operations": "Put one extra pilot boat and tug on standby. Check anchorage spacing.",
    },
    "High": {
        "Port authority": "Prioritise berths for ships that have waited longest. Coordinate arrival slots with terminals. Open overflow anchorage.",
        "Shipping company": "Slow steam now rather than wait at anchor. Warn charterers about demurrage risk. Send early delay alerts.",
        "Logistics planner": "Delay truck bookings, move containers to alternative days, alert customers to late cargo.",
        "Marine operations": "Roster extra pilots and tugs. Increase anchorage monitoring and long-stay vessel checks.",
    },
}
def decision_table(q1, q2):
    rule = {"Low": f"queue <= {q1:.0f} ships", "Medium": f"queue {q1 + 1:.0f}-{q2:.0f} ships", "High": f"queue > {q2:.0f} ships"}
    rows = [dict(congestion=lvl, trigger=rule[lvl], client=c, action=ACTIONS[lvl][c])
            for lvl in ("Low", "Medium", "High") for c in CLIENTS]
    return pd.DataFrame(rows)
