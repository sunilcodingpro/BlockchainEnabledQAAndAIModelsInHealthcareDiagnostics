"""
Demo Script for Blockchain-Enabled QA and AI Models in Healthcare Diagnostics

This script demonstrates the main components:
- Model registration
- Data provenance logging
- Decision auditing
- Regulatory report generation
"""

import os
import json
from framework.blockchain.hyperledger_connector import HyperledgerConnector
from framework.model_registry.registry import ModelRegistry
from framework.data_provenance.provenance_logger import DataProvenanceLogger
from framework.decision_audit.audit_logger import DecisionAuditLogger
from framework.regulatory.report_generator import RegulatoryReportGenerator

# --- Setup framework components ---
blockchain = HyperledgerConnector(
    config_path="network.yaml",
    channel_name="qahealthchannel",
    chaincode_name="aiqa_cc",
    org_name="HealthcareOrg",
    user_name="admin"
)
model_registry = ModelRegistry(blockchain)
data_logger = DataProvenanceLogger(blockchain)
audit_logger = DecisionAuditLogger(blockchain)
report_generator = RegulatoryReportGenerator(blockchain)

# --- Demo variables ---
MODEL_NAME = "CardioNet_v1"
MODEL_PATH = "models/cardionet_v1.bin"
MODEL_METADATA = {"accuracy": 0.95, "date": "2025-09-01"}
SAMPLE_ID = "sample_123"
SAMPLE_DATA = {"age": 60, "bp": 120, "cholesterol": 180}
SAMPLE_PROVENANCE = {"source": "Hospital A", "date": "2025-08-30"}
CASE_ID = "case_001"
DECISION = "diagnosis: healthy"
EXPLANATION = {"feature_importance": {"age": 0.2, "bp": 0.1, "cholesterol": 0.7}}

# --- Ensure model file exists (simulate) ---
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
with open(MODEL_PATH, "wb") as f:
    f.write(os.urandom(1024))  # Dummy model content

print("=== 1. Registering model ===")
model_hash = model_registry.register_model(MODEL_NAME, MODEL_PATH, MODEL_METADATA)
print(f"Model registered with hash: {model_hash}")

print("\n=== 2. Logging data provenance ===")
sample_hash = data_logger.log_sample(SAMPLE_ID, SAMPLE_DATA, SAMPLE_PROVENANCE)
print(f"Data sample logged with hash: {sample_hash}")

print("\n=== 3. Logging AI decision ===")
audit_logger.log_decision(CASE_ID, SAMPLE_DATA, MODEL_NAME, DECISION, EXPLANATION)
print("Decision logged.")

print("\n=== 4. Generating regulatory model report ===")
model_report_path = report_generator.generate_model_report(MODEL_NAME)
print(f"Model report generated at: {model_report_path}")

print("\n=== 5. Generating decision audit report ===")
decision_report_path = report_generator.generate_decision_audit_report(CASE_ID)
print(f"Decision audit report generated at: {decision_report_path}")

# Optional: Print report contents
for report_path in [model_report_path, decision_report_path]:
    print(f"\n--- Report: {report_path} ---")
    with open(report_path, 'r') as f:
        print(f.read())
