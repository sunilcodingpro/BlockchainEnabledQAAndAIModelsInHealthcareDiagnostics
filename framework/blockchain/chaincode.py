"""
Hyperledger Fabric Chaincode for AI/ML Healthcare Diagnostics QA Framework

This chaincode provides comprehensive blockchain functionality for:
- Model registry and versioning
- Diagnostic data logging and audit trails  
- Drift detection and compliance monitoring
- Quality assurance workflows
"""

import json
import hashlib
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class ModelMetadata:
    """Metadata for registered AI/ML models"""
    model_id: str
    name: str
    version: str
    algorithm: str
    accuracy: float
    training_date: str
    validation_metrics: Dict[str, float]
    regulatory_status: str
    creator_org: str
    model_hash: str
    created_at: str


@dataclass
class DiagnosticRecord:
    """Record of AI diagnostic operation"""
    diagnostic_id: str
    model_id: str
    patient_id: str  # Anonymized
    input_features: Dict[str, Any]
    prediction: Dict[str, Any]
    confidence_score: float
    explanation: Dict[str, Any]
    timestamp: str
    diagnostician_id: str
    audit_trail: List[str]


@dataclass
class ComplianceEvent:
    """Compliance monitoring event"""
    event_id: str
    event_type: str  # drift_detected, accuracy_drop, regulatory_violation
    model_id: str
    severity: str  # low, medium, high, critical
    description: str
    metrics: Dict[str, float]
    timestamp: str
    resolved: bool
    resolution_notes: str


class HealthcareDiagnosticsChaincode:
    """
    Comprehensive chaincode for healthcare AI diagnostics quality assurance
    
    Provides functions for model registry, diagnostic logging, drift detection,
    and compliance monitoring with full audit trails.
    """
    
    def __init__(self, ctx):
        """Initialize chaincode with fabric context"""
        self.ctx = ctx
        
    # === Model Registry Functions ===
    
    async def register_model(self, model_data: str) -> str:
        """
        Register a new AI/ML model in the blockchain registry
        
        Args:
            model_data: JSON string containing model metadata
            
        Returns:
            Transaction ID for the registration
        """
        try:
            data = json.loads(model_data)
            
            # Create model hash from content and metadata
            model_content = data.get('model_content', '')
            metadata_str = json.dumps(data.get('metadata', {}), sort_keys=True)
            model_hash = hashlib.sha256(
                (model_content + metadata_str).encode()
            ).hexdigest()
            
            # Create model metadata record
            model = ModelMetadata(
                model_id=data['model_id'],
                name=data['name'],
                version=data['version'],
                algorithm=data['algorithm'],
                accuracy=data['accuracy'],
                training_date=data['training_date'],
                validation_metrics=data['validation_metrics'],
                regulatory_status=data.get('regulatory_status', 'pending'),
                creator_org=data['creator_org'],
                model_hash=model_hash,
                created_at=datetime.utcnow().isoformat()
            )
            
            # Store in blockchain state
            model_key = f"model:{model.model_id}"
            await self.ctx.stub.put_state(model_key, json.dumps(asdict(model)))
            
            # Create index for queries
            await self._create_model_index(model)
            
            return model.model_id
            
        except Exception as e:
            raise Exception(f"Failed to register model: {str(e)}")
    
    async def get_model(self, model_id: str) -> str:
        """Get model metadata by ID"""
        model_key = f"model:{model_id}"
        model_data = await self.ctx.stub.get_state(model_key)
        
        if not model_data:
            raise Exception(f"Model {model_id} not found")
            
        return model_data.decode('utf-8')
    
    async def update_model_status(self, model_id: str, status: str, notes: str = "") -> str:
        """Update regulatory/approval status of a model"""
        model_data = await self.get_model(model_id)
        model = json.loads(model_data)
        
        model['regulatory_status'] = status
        if notes:
            if 'status_history' not in model:
                model['status_history'] = []
            model['status_history'].append({
                'status': status,
                'notes': notes,
                'timestamp': datetime.utcnow().isoformat()
            })
        
        model_key = f"model:{model_id}"
        await self.ctx.stub.put_state(model_key, json.dumps(model))
        
        return f"Model {model_id} status updated to {status}"
    
    # === Diagnostic Logging Functions ===
    
    async def submit_diagnostic(self, diagnostic_data: str) -> str:
        """
        Log an AI diagnostic operation to the blockchain
        
        Args:
            diagnostic_data: JSON string with diagnostic details
            
        Returns:
            Diagnostic record ID
        """
        try:
            data = json.loads(diagnostic_data)
            
            diagnostic = DiagnosticRecord(
                diagnostic_id=data['diagnostic_id'],
                model_id=data['model_id'],
                patient_id=data['patient_id'],  # Should be anonymized
                input_features=data['input_features'],
                prediction=data['prediction'],
                confidence_score=data['confidence_score'],
                explanation=data.get('explanation', {}),
                timestamp=datetime.utcnow().isoformat(),
                diagnostician_id=data['diagnostician_id'],
                audit_trail=[f"Diagnostic created at {datetime.utcnow().isoformat()}"]
            )
            
            # Verify model exists and is approved
            await self._verify_model_approved(diagnostic.model_id)
            
            # Store diagnostic record
            diag_key = f"diagnostic:{diagnostic.diagnostic_id}"
            await self.ctx.stub.put_state(diag_key, json.dumps(asdict(diagnostic)))
            
            # Create indexes for queries
            await self._create_diagnostic_index(diagnostic)
            
            # Trigger drift detection
            await self._check_model_drift(diagnostic.model_id, diagnostic)
            
            return diagnostic.diagnostic_id
            
        except Exception as e:
            raise Exception(f"Failed to submit diagnostic: {str(e)}")
    
    async def get_diagnostic(self, diagnostic_id: str) -> str:
        """Get diagnostic record by ID"""
        diag_key = f"diagnostic:{diagnostic_id}"
        diag_data = await self.ctx.stub.get_state(diag_key)
        
        if not diag_data:
            raise Exception(f"Diagnostic {diagnostic_id} not found")
            
        return diag_data.decode('utf-8')
    
    # === Audit Trail Functions ===
    
    async def get_audit_trail(self, model_id: str, start_date: str = "", end_date: str = "") -> str:
        """
        Get comprehensive audit trail for a model
        
        Args:
            model_id: Model identifier
            start_date: Optional start date filter (ISO format)
            end_date: Optional end date filter (ISO format)
            
        Returns:
            JSON array of audit records
        """
        try:
            # Get all diagnostics for this model
            diagnostics = await self._query_diagnostics_by_model(
                model_id, start_date, end_date
            )
            
            # Get compliance events
            compliance_events = await self._query_compliance_events_by_model(
                model_id, start_date, end_date
            )
            
            # Get model status changes
            model_data = await self.get_model(model_id)
            model = json.loads(model_data)
            status_history = model.get('status_history', [])
            
            audit_trail = {
                'model_id': model_id,
                'model_info': model,
                'diagnostics': diagnostics,
                'compliance_events': compliance_events,
                'status_history': status_history,
                'generated_at': datetime.utcnow().isoformat()
            }
            
            return json.dumps(audit_trail)
            
        except Exception as e:
            raise Exception(f"Failed to get audit trail: {str(e)}")
    
    # === Compliance and Drift Detection ===
    
    async def report_compliance_event(self, event_data: str) -> str:
        """Report a compliance violation or drift detection event"""
        try:
            data = json.loads(event_data)
            
            event = ComplianceEvent(
                event_id=data['event_id'],
                event_type=data['event_type'],
                model_id=data['model_id'],
                severity=data['severity'],
                description=data['description'],
                metrics=data.get('metrics', {}),
                timestamp=datetime.utcnow().isoformat(),
                resolved=False,
                resolution_notes=""
            )
            
            # Store compliance event
            event_key = f"compliance:{event.event_id}"
            await self.ctx.stub.put_state(event_key, json.dumps(asdict(event)))
            
            # Create index
            await self._create_compliance_index(event)
            
            # If critical, automatically flag model for review
            if event.severity == 'critical':
                await self.update_model_status(
                    event.model_id, 
                    'under_review',
                    f"Critical compliance event: {event.description}"
                )
            
            return event.event_id
            
        except Exception as e:
            raise Exception(f"Failed to report compliance event: {str(e)}")
    
    async def resolve_compliance_event(self, event_id: str, resolution_notes: str) -> str:
        """Mark a compliance event as resolved"""
        event_key = f"compliance:{event_id}"
        event_data = await self.ctx.stub.get_state(event_key)
        
        if not event_data:
            raise Exception(f"Compliance event {event_id} not found")
        
        event = json.loads(event_data.decode('utf-8'))
        event['resolved'] = True
        event['resolution_notes'] = resolution_notes
        event['resolved_at'] = datetime.utcnow().isoformat()
        
        await self.ctx.stub.put_state(event_key, json.dumps(event))
        
        return f"Compliance event {event_id} resolved"
    
    async def generate_compliance_report(self, model_id: str = "", org_id: str = "") -> str:
        """Generate comprehensive compliance report"""
        try:
            filters = {}
            if model_id:
                filters['model_id'] = model_id
            if org_id:
                filters['org_id'] = org_id
            
            # Get all relevant compliance events
            all_events = await self._query_all_compliance_events(filters)
            
            # Aggregate statistics
            total_events = len(all_events)
            resolved_events = len([e for e in all_events if e.get('resolved', False)])
            critical_events = len([e for e in all_events if e.get('severity') == 'critical'])
            
            report = {
                'report_id': hashlib.sha256(
                    f"{model_id}{org_id}{datetime.utcnow()}".encode()
                ).hexdigest()[:16],
                'generated_at': datetime.utcnow().isoformat(),
                'scope': {'model_id': model_id, 'org_id': org_id},
                'summary': {
                    'total_events': total_events,
                    'resolved_events': resolved_events,
                    'unresolved_events': total_events - resolved_events,
                    'critical_events': critical_events,
                    'compliance_score': (resolved_events / total_events * 100) if total_events > 0 else 100
                },
                'events': all_events
            }
            
            return json.dumps(report)
            
        except Exception as e:
            raise Exception(f"Failed to generate compliance report: {str(e)}")
    
    # === Private Helper Functions ===
    
    async def _verify_model_approved(self, model_id: str):
        """Verify that a model is approved for use"""
        model_data = await self.get_model(model_id)
        model = json.loads(model_data)
        
        if model['regulatory_status'] not in ['approved', 'conditionally_approved']:
            raise Exception(f"Model {model_id} not approved for diagnostic use")
    
    async def _check_model_drift(self, model_id: str, diagnostic: DiagnosticRecord):
        """Check for model drift based on recent diagnostics"""
        # This would implement statistical drift detection
        # For now, implement basic confidence threshold check
        
        if diagnostic.confidence_score < 0.7:  # Threshold for concern
            event_data = json.dumps({
                'event_id': f"drift_{model_id}_{int(time.time())}",
                'event_type': 'low_confidence',
                'model_id': model_id,
                'severity': 'medium' if diagnostic.confidence_score < 0.5 else 'low',
                'description': f"Low confidence prediction: {diagnostic.confidence_score}",
                'metrics': {'confidence_score': diagnostic.confidence_score}
            })
            
            await self.report_compliance_event(event_data)
    
    async def _create_model_index(self, model: ModelMetadata):
        """Create composite key indexes for model queries"""
        # Index by organization
        org_key = self.ctx.stub.create_composite_key('model~org', [model.creator_org, model.model_id])
        await self.ctx.stub.put_state(org_key, b'\x00')
        
        # Index by status
        status_key = self.ctx.stub.create_composite_key('model~status', [model.regulatory_status, model.model_id])
        await self.ctx.stub.put_state(status_key, b'\x00')
    
    async def _create_diagnostic_index(self, diagnostic: DiagnosticRecord):
        """Create indexes for diagnostic queries"""
        # Index by model
        model_key = self.ctx.stub.create_composite_key('diag~model', [diagnostic.model_id, diagnostic.diagnostic_id])
        await self.ctx.stub.put_state(model_key, b'\x00')
        
        # Index by date
        date_key = diagnostic.timestamp[:10]  # YYYY-MM-DD
        date_index = self.ctx.stub.create_composite_key('diag~date', [date_key, diagnostic.diagnostic_id])
        await self.ctx.stub.put_state(date_index, b'\x00')
    
    async def _create_compliance_index(self, event: ComplianceEvent):
        """Create indexes for compliance event queries"""
        # Index by model
        model_key = self.ctx.stub.create_composite_key('compliance~model', [event.model_id, event.event_id])
        await self.ctx.stub.put_state(model_key, b'\x00')
        
        # Index by severity
        severity_key = self.ctx.stub.create_composite_key('compliance~severity', [event.severity, event.event_id])
        await self.ctx.stub.put_state(severity_key, b'\x00')
    
    async def _query_diagnostics_by_model(self, model_id: str, start_date: str = "", end_date: str = "") -> List[Dict]:
        """Query diagnostics for a specific model"""
        # Implementation would use composite key queries
        # This is a simplified version
        diagnostics = []
        
        iterator = await self.ctx.stub.get_state_by_partial_composite_key('diag~model', [model_id])
        
        async for key, value in iterator:
            _, _, diag_id = self.ctx.stub.split_composite_key(key)
            diag_data = await self.get_diagnostic(diag_id)
            diagnostic = json.loads(diag_data)
            
            # Apply date filters if provided
            if start_date and diagnostic['timestamp'] < start_date:
                continue
            if end_date and diagnostic['timestamp'] > end_date:
                continue
                
            diagnostics.append(diagnostic)
        
        return diagnostics
    
    async def _query_compliance_events_by_model(self, model_id: str, start_date: str = "", end_date: str = "") -> List[Dict]:
        """Query compliance events for a specific model"""
        events = []
        
        iterator = await self.ctx.stub.get_state_by_partial_composite_key('compliance~model', [model_id])
        
        async for key, value in iterator:
            _, _, event_id = self.ctx.stub.split_composite_key(key)
            event_key = f"compliance:{event_id}"
            event_data = await self.ctx.stub.get_state(event_key)
            
            if event_data:
                event = json.loads(event_data.decode('utf-8'))
                
                # Apply date filters
                if start_date and event['timestamp'] < start_date:
                    continue
                if end_date and event['timestamp'] > end_date:
                    continue
                    
                events.append(event)
        
        return events
    
    async def _query_all_compliance_events(self, filters: Dict = None) -> List[Dict]:
        """Query all compliance events with optional filters"""
        events = []
        
        # This would implement more sophisticated querying
        # For now, return a placeholder
        return events


# === Chaincode Entry Points ===

def main():
    """Main chaincode entry point for Hyperledger Fabric"""
    pass  # This would be implemented according to Fabric SDK requirements

# Transaction functions that would be called by the Fabric runtime
async def invoke_register_model(ctx, model_data: str):
    """Fabric invoke function for model registration"""
    cc = HealthcareDiagnosticsChaincode(ctx)
    return await cc.register_model(model_data)

async def invoke_submit_diagnostic(ctx, diagnostic_data: str):
    """Fabric invoke function for diagnostic submission"""
    cc = HealthcareDiagnosticsChaincode(ctx)
    return await cc.submit_diagnostic(diagnostic_data)

async def query_audit_trail(ctx, model_id: str, start_date: str = "", end_date: str = ""):
    """Fabric query function for audit trail"""
    cc = HealthcareDiagnosticsChaincode(ctx)
    return await cc.get_audit_trail(model_id, start_date, end_date)

async def query_compliance_report(ctx, model_id: str = "", org_id: str = ""):
    """Fabric query function for compliance report"""
    cc = HealthcareDiagnosticsChaincode(ctx)
    return await cc.generate_compliance_report(model_id, org_id)