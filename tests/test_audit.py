from framework.decision_audit.audit import DecisionAudit

def test_audit_log_and_retrieve():
    audit = DecisionAudit()
    entry = audit.log("model-1", {"input": 42}, "output", "test_user")
    assert entry["status"] == "logged"
    logs = audit.get_audit_trail("model-1")
    assert len(logs) == 1
    assert logs[0]["user"] == "test_user"
