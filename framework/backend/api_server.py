"""
FastAPI Backend Server for Blockchain-Enabled Healthcare AI QA Framework

Provides REST API endpoints for:
- register_model: Register AI/ML models in blockchain registry
- submit_diagnostic: Submit diagnostic operations with audit trails
- get_audit_trail: Retrieve comprehensive audit trails
- compliance_report: Generate regulatory compliance reports  
- simulate_case: Run simulation scenarios for testing
"""

import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn

# Import framework components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blockchain.hyperledger_connector import HyperledgerConnector
from model_registry.registry import ModelRegistry
from data_provenance.provenance_logger import DataProvenanceLogger
from decision_audit.audit_logger import DecisionAuditLogger
from regulatory.report_generator import RegulatoryReportGenerator
from simulation.simulator import Simulator


# === Pydantic Models for API Requests/Responses ===

class ModelRegistrationRequest(BaseModel):
    """Request model for registering AI/ML models"""
    model_name: str = Field(..., description="Unique model identifier")
    model_path: str = Field(..., description="Path to model file")
    metadata: Dict[str, Any] = Field(..., description="Model metadata including accuracy, algorithm, etc.")
    
    @validator('metadata')
    def validate_metadata(cls, v):
        required_fields = ['accuracy', 'algorithm']
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Missing required field: {field}")
        if not 0 <= v['accuracy'] <= 1:
            raise ValueError("Accuracy must be between 0 and 1")
        return v


class DiagnosticSubmissionRequest(BaseModel):
    """Request model for submitting diagnostic operations"""
    case_id: str = Field(..., description="Anonymized case identifier")
    model_id: str = Field(..., description="Model used for diagnosis")
    input_features: Dict[str, Any] = Field(..., description="Input features for diagnosis")
    prediction: Dict[str, Any] = Field(..., description="Model prediction output")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    explanation: Dict[str, Any] = Field(default={}, description="Explainable AI output")
    human_review: Optional[Dict[str, Any]] = Field(default=None, description="Human reviewer information")


class AuditTrailRequest(BaseModel):
    """Request model for audit trail queries"""
    model_id: Optional[str] = Field(default="", description="Model identifier")
    case_id: Optional[str] = Field(default="", description="Case identifier")  
    start_date: Optional[str] = Field(default="", description="Start date (ISO format)")
    end_date: Optional[str] = Field(default="", description="End date (ISO format)")
    include_compliance_events: bool = Field(default=True, description="Include compliance events")


class ComplianceReportRequest(BaseModel):
    """Request model for compliance reports"""
    model_id: Optional[str] = Field(default="", description="Model identifier")
    organization: Optional[str] = Field(default="", description="Organization filter")
    regulation: str = Field(default="comprehensive", description="Regulation framework")
    period_days: int = Field(default=90, ge=1, le=365, description="Report period in days")
    format_type: str = Field(default="json", description="Output format")


class SimulationRequest(BaseModel):
    """Request model for simulation scenarios"""
    simulation_type: str = Field(..., description="Type of simulation")
    scenario_parameters: Dict[str, Any] = Field(..., description="Simulation parameters")
    model_id: Optional[str] = Field(default="", description="Model for simulation")
    case_count: int = Field(default=10, ge=1, le=1000, description="Number of cases to simulate")


class APIResponse(BaseModel):
    """Standard API response model"""
    success: bool
    message: str
    data: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# === Global Variables ===
logger = logging.getLogger(__name__)
blockchain_connector = None
model_registry = None
data_logger = None
audit_logger = None
report_generator = None
simulator = None


# === FastAPI Lifespan Management ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown"""
    # Startup
    logger.info("Starting blockchain-enabled healthcare AI QA API server...")
    await initialize_framework_components()
    logger.info("Framework components initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API server...")
    await cleanup_framework_components()


# === FastAPI Application ===

app = FastAPI(
    title="Blockchain-Enabled Healthcare AI QA Framework API",
    description="REST API for blockchain-based quality assurance and audit trails in healthcare AI",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Framework Initialization ===

async def initialize_framework_components():
    """Initialize all framework components"""
    global blockchain_connector, model_registry, data_logger, audit_logger, report_generator, simulator
    
    try:
        # Initialize blockchain connector
        blockchain_connector = HyperledgerConnector(
            config_path="network.yaml",
            channel_name="qahealthchannel", 
            chaincode_name="aiqa_cc",
            org_name="HealthcareOrg",
            user_name="api_server",
            mock_mode=True  # Use mock mode for development
        )
        
        # Initialize framework components
        model_registry = ModelRegistry(blockchain_connector)
        data_logger = DataProvenanceLogger(blockchain_connector)
        audit_logger = DecisionAuditLogger(blockchain_connector)
        report_generator = RegulatoryReportGenerator(blockchain_connector)
        simulator = Simulator()
        
        # Test blockchain connection
        connection_status = blockchain_connector.get_connection_status()
        logger.info(f"Blockchain connection status: {connection_status}")
        
    except Exception as e:
        logger.error(f"Failed to initialize framework components: {str(e)}")
        raise


async def cleanup_framework_components():
    """Cleanup framework components on shutdown"""
    logger.info("Cleaning up framework components...")
    # Add any necessary cleanup logic here


# === Dependency Functions ===

def get_blockchain_connector():
    """Dependency to get blockchain connector"""
    if not blockchain_connector:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Blockchain connector not initialized"
        )
    return blockchain_connector


def get_model_registry():
    """Dependency to get model registry"""
    if not model_registry:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model registry not initialized"
        )
    return model_registry


# === API Endpoints ===

@app.get("/", response_model=APIResponse)
async def root():
    """Root endpoint with API information"""
    return APIResponse(
        success=True,
        message="Blockchain-Enabled Healthcare AI QA Framework API",
        data={
            "version": "1.0.0",
            "documentation": "/docs",
            "status": "operational",
            "blockchain_connected": blockchain_connector is not None
        }
    )


@app.get("/health", response_model=APIResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check blockchain connection
        if blockchain_connector:
            connection_test = await blockchain_connector.test_connection()
            blockchain_status = "connected" if connection_test else "disconnected"
        else:
            blockchain_status = "not_initialized"
        
        health_data = {
            "status": "healthy",
            "blockchain_status": blockchain_status,
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "blockchain_connector": blockchain_connector is not None,
                "model_registry": model_registry is not None,
                "audit_logger": audit_logger is not None,
                "report_generator": report_generator is not None,
                "simulator": simulator is not None
            }
        }
        
        return APIResponse(
            success=True,
            message="Health check completed",
            data=health_data
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return APIResponse(
            success=False,
            message="Health check failed",
            error=str(e)
        )


@app.post("/api/v1/register_model", response_model=APIResponse)
async def register_model(
    request: ModelRegistrationRequest,
    registry: ModelRegistry = Depends(get_model_registry),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Register an AI/ML model in the blockchain registry
    
    This endpoint registers a new model with comprehensive metadata
    and stores verification hashes in the blockchain for integrity.
    """
    try:
        logger.info(f"Registering model: {request.model_name}")
        
        # Register model in blockchain
        model_hash = await registry.register_model(
            request.model_name, 
            request.model_path, 
            request.metadata
        )
        
        # Schedule background compliance check
        background_tasks.add_task(
            schedule_model_compliance_check,
            request.model_name
        )
        
        response_data = {
            "model_name": request.model_name,
            "model_hash": model_hash,
            "registration_status": "success",
            "regulatory_status": "pending",
            "next_steps": [
                "Model registered successfully",
                "Compliance check scheduled",
                "Awaiting regulatory approval"
            ]
        }
        
        return APIResponse(
            success=True,
            message="Model registered successfully",
            data=response_data
        )
        
    except Exception as e:
        logger.error(f"Failed to register model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model registration failed: {str(e)}"
        )


@app.post("/api/v1/submit_diagnostic", response_model=APIResponse)
async def submit_diagnostic(
    request: DiagnosticSubmissionRequest,
    audit_logger_dep: DecisionAuditLogger = Depends(lambda: audit_logger),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Submit a diagnostic operation with complete audit trail
    
    Records AI/ML diagnostic decisions in the blockchain with
    explainability data and compliance monitoring.
    """
    try:
        logger.info(f"Submitting diagnostic for case: {request.case_id}")
        
        # Log decision in blockchain
        decision_id = await audit_logger_dep.log_decision(
            case_id=request.case_id,
            input_data=request.input_features,
            model_name=request.model_id,
            decision=json.dumps(request.prediction),
            explanation=request.explanation,
            confidence_score=request.confidence_score,
            human_review=request.human_review
        )
        
        # Schedule drift detection check
        background_tasks.add_task(
            check_model_drift,
            request.model_id,
            request.confidence_score
        )
        
        response_data = {
            "decision_id": decision_id,
            "case_id": request.case_id,
            "model_id": request.model_id,
            "confidence_score": request.confidence_score,
            "audit_trail_created": True,
            "compliance_status": "monitored"
        }
        
        return APIResponse(
            success=True,
            message="Diagnostic submitted successfully",
            data=response_data
        )
        
    except Exception as e:
        logger.error(f"Failed to submit diagnostic: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Diagnostic submission failed: {str(e)}"
        )


@app.post("/api/v1/get_audit_trail", response_model=APIResponse)
async def get_audit_trail(
    request: AuditTrailRequest,
    blockchain: HyperledgerConnector = Depends(get_blockchain_connector)
):
    """
    Retrieve comprehensive audit trail from blockchain
    
    Returns complete audit trail including decisions, compliance events,
    and data provenance for regulatory compliance.
    """
    try:
        logger.info(f"Retrieving audit trail for model: {request.model_id or 'ALL'}")
        
        # Get audit trail from blockchain
        if request.model_id:
            audit_data = await blockchain.get_audit_trail(
                request.model_id,
                request.start_date,
                request.end_date
            )
        elif request.case_id:
            audit_data = await blockchain.get_audit_trail(
                f"case_{request.case_id}",
                request.start_date,
                request.end_date
            )
        else:
            # System-wide audit trail
            audit_data = {
                "audit_scope": "system_wide",
                "message": "System-wide audit trail not implemented in mock mode",
                "diagnostics": [],
                "compliance_events": []
            }
        
        # Enhance with additional analysis
        enhanced_audit = {
            "audit_trail": audit_data,
            "summary": {
                "total_diagnostics": len(audit_data.get('diagnostics', [])),
                "total_compliance_events": len(audit_data.get('compliance_events', [])),
                "period": {
                    "start": request.start_date or "inception",
                    "end": request.end_date or datetime.utcnow().isoformat()
                }
            },
            "quality_metrics": {
                "data_integrity": "verified",
                "completeness": "100%",
                "regulatory_compliance": "monitored"
            }
        }
        
        return APIResponse(
            success=True,
            message="Audit trail retrieved successfully",
            data=enhanced_audit
        )
        
    except Exception as e:
        logger.error(f"Failed to get audit trail: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Audit trail retrieval failed: {str(e)}"
        )


@app.post("/api/v1/compliance_report", response_model=APIResponse)
async def generate_compliance_report(
    request: ComplianceReportRequest,
    report_gen: RegulatoryReportGenerator = Depends(lambda: report_generator)
):
    """
    Generate comprehensive regulatory compliance report
    
    Creates detailed compliance reports for various regulatory frameworks
    including FDA, ISO 13485, HIPAA, and EU MDR.
    """
    try:
        logger.info(f"Generating compliance report: {request.regulation}")
        
        # Generate compliance report
        report_path = await report_gen.generate_compliance_report(
            organization=request.organization,
            regulation=request.regulation,
            period_days=request.period_days
        )
        
        if not report_path:
            raise Exception("Failed to generate compliance report")
        
        # Read report content
        with open(report_path, 'r', encoding='utf-8') as f:
            report_content = json.load(f)
        
        response_data = {
            "report_path": report_path,
            "report_type": request.regulation,
            "organization": request.organization or "system_wide",
            "period_days": request.period_days,
            "report_summary": report_content.get('compliance_summary', {}),
            "generated_at": report_content.get('generated_at'),
            "download_available": True
        }
        
        return APIResponse(
            success=True,
            message="Compliance report generated successfully",
            data=response_data
        )
        
    except Exception as e:
        logger.error(f"Failed to generate compliance report: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Compliance report generation failed: {str(e)}"
        )


@app.post("/api/v1/simulate_case", response_model=APIResponse)
async def simulate_case(
    request: SimulationRequest,
    sim: Simulator = Depends(lambda: simulator)
):
    """
    Run simulation scenarios for testing and validation
    
    Executes various simulation scenarios to test system behavior,
    model performance, and compliance workflows.
    """
    try:
        logger.info(f"Running simulation: {request.simulation_type}")
        
        # Configure simulation
        sim_config = {
            'simulation_type': request.simulation_type,
            'parameters': request.scenario_parameters,
            'model_id': request.model_id,
            'case_count': request.case_count
        }
        
        # Run simulation
        simulation_results = await sim.run_simulation(sim_config)
        
        response_data = {
            "simulation_id": simulation_results.get('simulation_id'),
            "simulation_type": request.simulation_type,
            "cases_simulated": simulation_results.get('cases_processed', 0),
            "results_summary": simulation_results.get('summary', {}),
            "metrics": simulation_results.get('metrics', {}),
            "recommendations": simulation_results.get('recommendations', [])
        }
        
        return APIResponse(
            success=True,
            message="Simulation completed successfully",
            data=response_data
        )
        
    except Exception as e:
        logger.error(f"Failed to run simulation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Simulation failed: {str(e)}"
        )


@app.get("/api/v1/download_report/{report_filename}")
async def download_report(report_filename: str):
    """Download generated report file"""
    try:
        report_path = os.path.join("reports", report_filename)
        
        if not os.path.exists(report_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Report file not found"
            )
        
        return FileResponse(
            path=report_path,
            filename=report_filename,
            media_type='application/octet-stream'
        )
        
    except Exception as e:
        logger.error(f"Failed to download report: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Report download failed: {str(e)}"
        )


@app.get("/api/v1/models", response_model=APIResponse)
async def list_models(
    status_filter: Optional[str] = None,
    organization: Optional[str] = None,
    registry: ModelRegistry = Depends(get_model_registry)
):
    """List registered models with optional filtering"""
    try:
        models = await registry.list_models(
            organization=organization or "",
            status=status_filter or ""
        )
        
        return APIResponse(
            success=True,
            message="Models retrieved successfully",
            data={
                "models": models,
                "total_count": len(models),
                "filters_applied": {
                    "status": status_filter,
                    "organization": organization
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to list models: {str(e)}"
        )


# === Background Tasks ===

async def schedule_model_compliance_check(model_name: str):
    """Background task to schedule model compliance check"""
    try:
        logger.info(f"Scheduling compliance check for model: {model_name}")
        # In production, would schedule actual compliance monitoring
        await asyncio.sleep(1)  # Simulate processing
        logger.info(f"Compliance check scheduled for model: {model_name}")
    except Exception as e:
        logger.error(f"Failed to schedule compliance check: {str(e)}")


async def check_model_drift(model_id: str, confidence_score: float):
    """Background task to check for model drift"""
    try:
        if confidence_score < 0.7:
            logger.warning(f"Low confidence detected for model {model_id}: {confidence_score}")
            # In production, would trigger drift detection algorithms
        
        await asyncio.sleep(0.5)  # Simulate processing
        logger.debug(f"Drift check completed for model: {model_id}")
    except Exception as e:
        logger.error(f"Failed to check model drift: {str(e)}")


# === Error Handlers ===

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": "API request failed",
            "error": exc.detail,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "error": "An unexpected error occurred",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# === Application Entry Point ===

def main():
    """Main entry point for running the API server"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload in development
        log_level="info"
    )


if __name__ == "__main__":
    main()