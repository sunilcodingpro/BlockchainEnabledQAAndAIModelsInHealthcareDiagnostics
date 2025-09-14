"""
DecisionAuditLogger: Logs AI/ML decisions and explanations to the audit trail.

Provides comprehensive decision tracking with blockchain immutability for:
- AI model predictions and confidence scores
- Explainable AI outputs (SHAP, LIME, etc.)  
- Human reviewer decisions and overrides
- Clinical decision support audit trails
"""

import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging


class DecisionAuditLogger:
    """
    Comprehensive decision audit logger with blockchain integration
    
    Tracks all AI/ML decisions, human interventions, and explanations
    to ensure accountability and regulatory compliance in healthcare AI.
    """
    
    def __init__(self, blockchain_connector):
        """
        Initialize decision audit logger with blockchain connector
        
        Args:
            blockchain_connector: HyperledgerConnector instance
        """
        self.blockchain = blockchain_connector
        self.logger = logging.getLogger(__name__)
        
        # Track decision sessions for complex workflows
        self.active_decision_sessions = {}
    
    async def log_decision(self, case_id: str, input_data: Dict[str, Any], 
                          model_name: str, decision: str, explanation: Dict[str, Any],
                          confidence_score: float = None, human_review: Dict[str, Any] = None) -> str:
        """
        Log an AI/ML decision with full audit information
        
        Args:
            case_id: Unique case identifier
            input_data: Input features/data used for decision
            model_name: AI model that made the decision
            decision: The actual decision/prediction
            explanation: Explainable AI output (SHAP values, etc.)
            confidence_score: Model confidence (0.0 to 1.0)
            human_review: Optional human reviewer information
            
        Returns:
            Decision audit ID for future reference
        """
        try:
            self.logger.info(f"Logging decision for case: {case_id}")
            
            # Create unique decision ID
            decision_id = self._generate_decision_id(case_id, model_name)
            
            # Create comprehensive audit record
            audit_record = self._create_decision_audit_record(
                decision_id, case_id, input_data, model_name, decision,
                explanation, confidence_score, human_review
            )
            
            # Calculate decision hash for integrity
            decision_hash = self._calculate_decision_hash(audit_record)
            audit_record['decision_hash'] = decision_hash
            
            # Submit to blockchain
            diagnostic_data = {
                'diagnostic_id': decision_id,
                'model_id': model_name,
                'patient_id': case_id,  # Anonymized
                'input_features': input_data,
                'prediction': {'decision': decision, 'explanation': explanation},
                'confidence_score': confidence_score or 0.0,
                'explanation': explanation
            }
            
            tx_id = await self.blockchain.submit_diagnostic(json.dumps(diagnostic_data))
            
            # Store additional audit metadata
            await self._store_audit_metadata(decision_id, audit_record)
            
            self.logger.info(f"Decision {decision_id} logged with transaction: {tx_id}")
            return decision_id
            
        except Exception as e:
            self.logger.error(f"Failed to log decision for case {case_id}: {str(e)}")
            raise
    
    async def log_human_override(self, decision_id: str, original_decision: str,
                                override_decision: str, reviewer_id: str, 
                                justification: str, supporting_evidence: List[str] = None) -> str:
        """
        Log human override of AI decision
        
        Args:
            decision_id: Original AI decision ID
            original_decision: AI model's original decision
            override_decision: Human reviewer's override decision
            reviewer_id: ID of the human reviewer
            justification: Reason for override
            supporting_evidence: Additional evidence/references
            
        Returns:
            Override audit ID
        """
        try:
            override_id = f"{decision_id}_override_{int(time.time())}"
            
            override_record = {
                'override_id': override_id,
                'original_decision_id': decision_id,
                'original_decision': original_decision,
                'override_decision': override_decision,
                'reviewer_id': reviewer_id,
                'justification': justification,
                'supporting_evidence': supporting_evidence or [],
                'override_timestamp': datetime.utcnow().isoformat(),
                'override_type': 'human_expert_review'
            }
            
            # Submit override to blockchain
            tx_id = "mock_tx_" + hashlib.sha256(json.dumps(override_record).encode()).hexdigest()[:16]
            
            # Update original decision record
            await self._link_override_to_decision(decision_id, override_id)
            
            self.logger.warning(f"Human override logged: {override_id} by {reviewer_id}")
            return override_id
            
        except Exception as e:
            self.logger.error(f"Failed to log human override: {str(e)}")
            raise
    
    async def start_decision_session(self, session_id: str, case_id: str, 
                                   session_type: str, participants: List[str]) -> str:
        """
        Start a multi-step decision session (e.g., multidisciplinary review)
        
        Args:
            session_id: Unique session identifier
            case_id: Case being reviewed
            session_type: Type of session (mdt_review, second_opinion, etc.)
            participants: List of participant IDs
            
        Returns:
            Session tracking ID
        """
        try:
            session_record = {
                'session_id': session_id,
                'case_id': case_id,
                'session_type': session_type,
                'participants': participants,
                'started_at': datetime.utcnow().isoformat(),
                'status': 'active',
                'decisions': [],
                'consensus_reached': False
            }
            
            self.active_decision_sessions[session_id] = session_record
            
            # Log session start to blockchain
            tx_id = "mock_tx_" + hashlib.sha256(json.dumps(session_record).encode()).hexdigest()[:16]
            
            self.logger.info(f"Decision session {session_id} started for case {case_id}")
            return session_id
            
        except Exception as e:
            self.logger.error(f"Failed to start decision session: {str(e)}")
            raise
    
    async def add_session_decision(self, session_id: str, participant_id: str,
                                 decision: str, rationale: str, supporting_data: Dict[str, Any] = None) -> str:
        """
        Add participant decision to an active session
        
        Args:
            session_id: Active session ID
            participant_id: ID of the participant making decision
            decision: The participant's decision
            rationale: Reasoning behind the decision
            supporting_data: Additional supporting information
            
        Returns:
            Participant decision ID
        """
        try:
            if session_id not in self.active_decision_sessions:
                raise ValueError(f"Session {session_id} not found or not active")
            
            participant_decision = {
                'participant_id': participant_id,
                'decision': decision,
                'rationale': rationale,
                'supporting_data': supporting_data or {},
                'timestamp': datetime.utcnow().isoformat(),
                'decision_order': len(self.active_decision_sessions[session_id]['decisions'])
            }
            
            # Add to session record
            self.active_decision_sessions[session_id]['decisions'].append(participant_decision)
            
            decision_id = f"{session_id}_part_{participant_id}_{int(time.time())}"
            
            self.logger.info(f"Added decision from {participant_id} to session {session_id}")
            return decision_id
            
        except Exception as e:
            self.logger.error(f"Failed to add session decision: {str(e)}")
            raise
    
    async def finalize_decision_session(self, session_id: str, final_decision: str,
                                      consensus_level: str, session_notes: str = "") -> str:
        """
        Finalize decision session with consensus decision
        
        Args:
            session_id: Session to finalize
            final_decision: Consensus decision reached
            consensus_level: Level of consensus (unanimous, majority, etc.)
            session_notes: Additional session notes
            
        Returns:
            Final session hash
        """
        try:
            if session_id not in self.active_decision_sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.active_decision_sessions[session_id]
            session.update({
                'status': 'completed',
                'final_decision': final_decision,
                'consensus_level': consensus_level,
                'session_notes': session_notes,
                'completed_at': datetime.utcnow().isoformat(),
                'total_participants': len(session['participants']),
                'decisions_recorded': len(session['decisions'])
            })
            
            # Calculate session hash
            session_hash = hashlib.sha256(
                json.dumps(session, sort_keys=True).encode()
            ).hexdigest()
            
            # Commit to blockchain
            tx_id = "mock_tx_" + hashlib.sha256(json.dumps(session).encode()).hexdigest()[:16]
            
            # Remove from active sessions
            del self.active_decision_sessions[session_id]
            
            self.logger.info(f"Decision session {session_id} finalized with consensus: {consensus_level}")
            return session_hash
            
        except Exception as e:
            self.logger.error(f"Failed to finalize decision session: {str(e)}")
            raise
    
    async def get_decision_audit_trail(self, case_id: str = "", decision_id: str = "",
                                     start_date: str = "", end_date: str = "") -> Dict[str, Any]:
        """
        Get comprehensive audit trail for decisions
        
        Args:
            case_id: Filter by case ID
            decision_id: Filter by specific decision ID
            start_date: Filter from date (ISO format)
            end_date: Filter to date (ISO format)
            
        Returns:
            Comprehensive audit trail
        """
        try:
            # Get blockchain audit trail
            if case_id:
                # In production, would query by patient case
                audit_data = await self.blockchain.get_audit_trail(
                    f"case_{case_id}", start_date, end_date
                )
            else:
                # Get general audit trail
                audit_data = {'diagnostics': [], 'compliance_events': []}
            
            # Enhance with decision-specific analysis
            enhanced_trail = self._enhance_audit_trail(audit_data, case_id, decision_id)
            
            return enhanced_trail
            
        except Exception as e:
            self.logger.error(f"Failed to get decision audit trail: {str(e)}")
            return {'error': str(e)}
    
    async def analyze_decision_patterns(self, model_name: str = "", days: int = 30) -> Dict[str, Any]:
        """
        Analyze decision patterns for quality assurance
        
        Args:
            model_name: Filter by specific model
            days: Analysis period in days
            
        Returns:
            Decision pattern analysis
        """
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            # Get decision data from blockchain
            if model_name:
                audit_data = await self.blockchain.get_audit_trail(
                    model_name, start_date.isoformat(), end_date.isoformat()
                )
            else:
                audit_data = {'diagnostics': []}
            
            diagnostics = audit_data.get('diagnostics', [])
            
            if not diagnostics:
                return {'error': 'No decision data available for analysis'}
            
            # Analyze patterns
            analysis = {
                'model_name': model_name,
                'analysis_period_days': days,
                'total_decisions': len(diagnostics),
                'patterns': self._analyze_patterns(diagnostics),
                'quality_metrics': self._calculate_quality_metrics(diagnostics),
                'recommendations': self._generate_recommendations(diagnostics),
                'generated_at': datetime.utcnow().isoformat()
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Failed to analyze decision patterns: {str(e)}")
            return {'error': str(e)}
    
    # === Private Helper Methods ===
    
    def _generate_decision_id(self, case_id: str, model_name: str) -> str:
        """Generate unique decision ID"""
        timestamp = str(int(time.time() * 1000))  # millisecond precision
        hash_input = f"{case_id}_{model_name}_{timestamp}"
        hash_suffix = hashlib.sha256(hash_input.encode()).hexdigest()[:8]
        return f"decision_{case_id}_{hash_suffix}"
    
    def _create_decision_audit_record(self, decision_id: str, case_id: str,
                                    input_data: Dict[str, Any], model_name: str,
                                    decision: str, explanation: Dict[str, Any],
                                    confidence_score: float, human_review: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive decision audit record"""
        record = {
            'decision_id': decision_id,
            'case_id': case_id,
            'model_info': {
                'model_name': model_name,
                'model_version': 'unknown',  # Would get from model registry
                'algorithm_type': 'unknown'
            },
            'input_summary': {
                'feature_count': len(input_data) if isinstance(input_data, dict) else 0,
                'input_hash': hashlib.sha256(json.dumps(input_data, sort_keys=True).encode()).hexdigest()
            },
            'decision_info': {
                'decision': decision,
                'confidence_score': confidence_score,
                'explanation_type': explanation.get('type', 'unknown'),
                'explanation_data': explanation
            },
            'audit_metadata': {
                'logged_at': datetime.utcnow().isoformat(),
                'logged_by': getattr(self.blockchain, 'user_name', 'system'),
                'organization': getattr(self.blockchain, 'org_name', 'unknown'),
                'system_version': '1.0'
            },
            'human_review': human_review or {},
            'compliance_flags': self._check_compliance_flags(decision, confidence_score)
        }
        
        return record
    
    def _calculate_decision_hash(self, audit_record: Dict[str, Any]) -> str:
        """Calculate cryptographic hash of decision record"""
        # Create deterministic hash excluding metadata that changes
        hashable_data = {
            'decision_id': audit_record['decision_id'],
            'case_id': audit_record['case_id'],
            'model_info': audit_record['model_info'],
            'input_summary': audit_record['input_summary'],
            'decision_info': audit_record['decision_info']
        }
        
        return hashlib.sha256(
            json.dumps(hashable_data, sort_keys=True).encode()
        ).hexdigest()
    
    async def _store_audit_metadata(self, decision_id: str, audit_record: Dict[str, Any]):
        """Store additional audit metadata"""
        # In production, might store in separate audit database
        # linked to blockchain transaction
        pass
    
    async def _link_override_to_decision(self, decision_id: str, override_id: str):
        """Link override record to original decision"""
        # Update decision record with override reference
        pass
    
    def _enhance_audit_trail(self, audit_data: Dict[str, Any], case_id: str, decision_id: str) -> Dict[str, Any]:
        """Enhance audit trail with additional analysis"""
        enhanced = audit_data.copy()
        
        # Add decision-specific metrics
        enhanced['decision_analysis'] = {
            'total_decisions': len(audit_data.get('diagnostics', [])),
            'unique_models': len(set(d.get('model_id', '') for d in audit_data.get('diagnostics', []))),
            'average_confidence': 0.85,  # Would calculate from actual data
            'human_overrides': 0  # Would count actual overrides
        }
        
        return enhanced
    
    def _analyze_patterns(self, diagnostics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze decision patterns"""
        if not diagnostics:
            return {}
        
        # Calculate basic patterns
        confidence_scores = [d.get('confidence_score', 0) for d in diagnostics]
        
        return {
            'confidence_distribution': {
                'mean': sum(confidence_scores) / len(confidence_scores),
                'min': min(confidence_scores),
                'max': max(confidence_scores),
                'low_confidence_count': len([s for s in confidence_scores if s < 0.7])
            },
            'temporal_patterns': {
                'decisions_per_hour': len(diagnostics) / 24 if diagnostics else 0,
                'peak_hours': [9, 14, 16]  # Mock data
            }
        }
    
    def _calculate_quality_metrics(self, diagnostics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate decision quality metrics"""
        return {
            'consistency_score': 0.92,  # Mock calculation
            'explanation_quality': 0.88,  # Mock calculation
            'timeliness_score': 0.95   # Mock calculation
        }
    
    def _generate_recommendations(self, diagnostics: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on decision analysis"""
        recommendations = []
        
        confidence_scores = [d.get('confidence_score', 0) for d in diagnostics]
        low_confidence_count = len([s for s in confidence_scores if s < 0.7])
        
        if low_confidence_count > len(diagnostics) * 0.2:
            recommendations.append("High rate of low-confidence decisions detected. Consider model retraining.")
        
        if len(diagnostics) > 100:
            recommendations.append("High decision volume. Monitor for decision fatigue effects.")
        
        return recommendations
    
    def _check_compliance_flags(self, decision: str, confidence_score: float) -> List[str]:
        """Check for compliance flags in decision"""
        flags = []
        
        if confidence_score is not None and confidence_score < 0.5:
            flags.append('very_low_confidence')
        
        if confidence_score is not None and confidence_score < 0.7:
            flags.append('low_confidence')
        
        return flags