"""
Decision Audit: Logs and retrieves audit trails for model inferences.
"""

class DecisionAudit:
    def __init__(self):
        self.audits = []

    def log(self, model_id, input_data, output, user):
        entry = {"model_id": model_id, "input": input_data, "output": output, "user": user}
        self.audits.append(entry)
        return {"status": "logged", "entry": entry}

    def get_audit_trail(self, model_id):
        return [a for a in self.audits if a["model_id"] == model_id]
