#!/usr/bin/env python3
"""
Test script to demonstrate the complete simulation framework.
"""

import os
import sys

# Add framework to path
sys.path.insert(0, '/home/runner/work/BlockchainEnabledQAAndAIModelsInHealthcareDiagnostics/BlockchainEnabledQAAndAIModelsInHealthcareDiagnostics')

from framework.blockchain.hyperledger_connector import HyperledgerConnector
from framework.model_registry.registry import ModelRegistry
from framework.data_provenance.provenance_logger import DataProvenanceLogger
from framework.decision_audit.audit_logger import DecisionAuditLogger
from framework.regulatory.report_generator import RegulatoryReportGenerator
from framework.simulation.simulator import Simulator

def main():
    print("=== Blockchain-Enabled Healthcare AI Simulation ===\n")
    
    # Initialize framework components
    print("1. Initializing blockchain and framework components...")
    blockchain = HyperledgerConnector(
        config_path="network.yaml",
        channel_name="qahealthchannel", 
        chaincode_name="aiqa_cc",
        org_name="HealthcareOrg",
        user_name="admin"
    )
    
    model_registry = ModelRegistry(blockchain)
    provenance_logger = DataProvenanceLogger(blockchain)
    audit_logger = DecisionAuditLogger(blockchain)
    report_generator = RegulatoryReportGenerator(blockchain)
    
    # Initialize simulator
    simulator = Simulator(
        blockchain_connector=blockchain,
        model_registry=model_registry,
        provenance_logger=provenance_logger,
        audit_logger=audit_logger,
        report_generator=report_generator
    )
    
    print("✓ All components initialized successfully\n")
    
    # Run healthcare diagnostic simulation
    print("2. Running healthcare diagnostic simulation...")
    results = simulator.run("healthcare_diagnostic")
    
    if results.get("status") == "completed":
        print("✓ Simulation completed successfully!")
        print(f"Simulation ID: {results['simulation_id']}")
        print(f"Steps completed: {len(results['steps'])}")
        
        # Print step details
        for step in results['steps']:
            print(f"  Step {step['step']}: {step['description']}")
            if 'records' in step:
                print(f"    - Records: {step['records']}")
            if 'model_name' in step:
                print(f"    - Model: {step['model_name']}")
            if 'predictions' in step:
                print(f"    - Predictions: {step['predictions']}")
            if 'report_files' in step:
                print(f"    - Reports: {step['report_files']}")
        
        print(f"\nSimulation Summary:")
        summary = results.get('summary', {})
        for key, value in summary.items():
            print(f"  - {key}: {value}")
            
    else:
        print(f"✗ Simulation failed: {results.get('error', 'Unknown error')}")
    
    print("\n=== Simulation Complete ===")

if __name__ == "__main__":
    main()