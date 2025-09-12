"""
DecisionAuditLogger: Logs AI/ML decisions and explanations to the audit trail.
"""
import json
import hashlib
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime


class DecisionAuditLogger:
    """
    Logger for AI/ML decision audit trails on blockchain.
    
    This class tracks model predictions, decisions, explanations,
    and associated metadata for regulatory compliance and auditing.
    """
    
    def __init__(self, blockchain_connector):
        """
        Initialize the decision audit logger.
        
        Args:
            blockchain_connector: Instance of HyperledgerConnector for blockchain operations
        """
        self.blockchain = blockchain_connector
        self._logger = logging.getLogger(__name__)
    
    def log_decision(self, case_id: str, input_data: Dict[str, Any], 
                    model_name: str, decision: str, 
                    explanation: Optional[Dict[str, Any]] = None,
                    confidence: Optional[float] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Log an AI/ML decision with explanation to the audit trail.
        
        Args:
            case_id: Unique identifier for the case
            input_data: Input data used for the decision
            model_name: Name of the model that made the decision
            decision: The actual decision/prediction made
            explanation: Explanation data (feature importance, etc.)
            confidence: Confidence score of the decision (0-1)
            metadata: Additional metadata about the decision
            
        Returns:
            Decision hash if successful, None if failed
        """
        try:
            # Validate inputs
            if not case_id or not model_name or not decision:
                raise ValueError("Case ID, model name, and decision are required")
            
            # Prepare decision record
            decision_record = {
                'case_id': case_id,
                'model_name': model_name,
                'decision': decision,
                'input_data_hash': self._calculate_data_hash(input_data),
                'timestamp': datetime.now().isoformat(),
                'confidence': confidence,
                'explanation': explanation or {},
                'metadata': metadata or {}
            }
            
            # Calculate decision hash
            decision_hash = self._calculate_decision_hash(decision_record)
            decision_record['decision_hash'] = decision_hash
            
            self._logger.info(f"Logging decision for case {case_id} with model {model_name}")
            
            # Store decision on blockchain
            response = self.blockchain.invoke_chaincode(
                "logDecision",
                [case_id, decision_hash, json.dumps(decision_record)]
            )
            
            if response.success:
                self._logger.info(f"Successfully logged decision for case {case_id}")
                return decision_hash
            else:
                self._logger.error(f"Failed to log decision on blockchain: {response.data}")
                return None
                
        except Exception as e:
            self._logger.error(f"Error logging decision for case {case_id}: {e}")
            return None
    
    def get_decision(self, case_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve decision information for a specific case.
        
        Args:
            case_id: ID of the case to retrieve
            
        Returns:
            Decision information dictionary or None if not found
        """
        try:
            response = self.blockchain.query_chaincode("getDecision", [case_id])
            
            if response.success and response.data:
                return response.data
            else:
                self._logger.debug(f"Decision for case {case_id} not found")
                return None
                
        except Exception as e:
            self._logger.error(f"Error retrieving decision for case {case_id}: {e}")
            return None
    
    def get_model_decisions(self, model_name: str) -> List[Dict[str, Any]]:
        """
        Get all decisions made by a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of decision records
        """
        try:
            all_decisions = self.list_all_decisions()
            model_decisions = [d for d in all_decisions 
                             if d.get('model_name') == model_name]
            
            # Sort by timestamp (most recent first)
            model_decisions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            return model_decisions
            
        except Exception as e:
            self._logger.error(f"Error retrieving decisions for model {model_name}: {e}")
            return []
    
    def get_decision_history(self, case_id: str) -> List[Dict[str, Any]]:
        """
        Get the complete decision history for a case (if multiple decisions were made).
        
        Args:
            case_id: ID of the case
            
        Returns:
            List of decision records for the case
        """
        try:
            all_decisions = self.list_all_decisions()
            case_decisions = [d for d in all_decisions 
                            if d.get('case_id') == case_id]
            
            # Sort by timestamp (chronological order)
            case_decisions.sort(key=lambda x: x.get('timestamp', ''))
            return case_decisions
            
        except Exception as e:
            self._logger.error(f"Error retrieving decision history for case {case_id}: {e}")
            return []
    
    def audit_decision_integrity(self, case_id: str) -> Dict[str, Any]:
        """
        Audit the integrity of a decision record.
        
        Args:
            case_id: ID of the case to audit
            
        Returns:
            Audit result dictionary with integrity status and details
        """
        try:
            decision_record = self.get_decision(case_id)
            if not decision_record:
                return {
                    'case_id': case_id,
                    'integrity_status': 'FAILED',
                    'reason': 'Decision record not found',
                    'audit_timestamp': datetime.now().isoformat()
                }
            
            # Verify decision hash
            stored_hash = decision_record.get('decision_hash')
            
            # Recalculate hash (excluding the hash field itself)
            temp_record = {k: v for k, v in decision_record.items() 
                          if k != 'decision_hash'}
            calculated_hash = self._calculate_decision_hash(temp_record)
            
            integrity_status = 'PASSED' if stored_hash == calculated_hash else 'FAILED'
            
            audit_result = {
                'case_id': case_id,
                'integrity_status': integrity_status,
                'stored_hash': stored_hash,
                'calculated_hash': calculated_hash,
                'decision_timestamp': decision_record.get('timestamp'),
                'model_name': decision_record.get('model_name'),
                'audit_timestamp': datetime.now().isoformat()
            }
            
            if integrity_status == 'FAILED':
                audit_result['reason'] = 'Hash mismatch - decision may have been tampered with'
            
            self._logger.info(f"Decision integrity audit for case {case_id}: {integrity_status}")
            return audit_result
            
        except Exception as e:
            self._logger.error(f"Error auditing decision integrity for case {case_id}: {e}")
            return {
                'case_id': case_id,
                'integrity_status': 'ERROR',
                'reason': str(e),
                'audit_timestamp': datetime.now().isoformat()
            }
    
    def generate_audit_report(self, start_date: Optional[str] = None, 
                            end_date: Optional[str] = None,
                            model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive audit report for decisions.
        
        Args:
            start_date: Start date for report (ISO format)
            end_date: End date for report (ISO format)
            model_name: Optional model name filter
            
        Returns:
            Audit report dictionary
        """
        try:
            all_decisions = self.list_all_decisions()
            
            # Apply filters
            filtered_decisions = all_decisions
            
            if start_date:
                filtered_decisions = [d for d in filtered_decisions 
                                    if d.get('timestamp', '') >= start_date]
            
            if end_date:
                filtered_decisions = [d for d in filtered_decisions 
                                    if d.get('timestamp', '') <= end_date]
            
            if model_name:
                filtered_decisions = [d for d in filtered_decisions 
                                    if d.get('model_name') == model_name]
            
            # Generate statistics
            total_decisions = len(filtered_decisions)
            unique_models = len(set(d.get('model_name') for d in filtered_decisions))
            unique_cases = len(set(d.get('case_id') for d in filtered_decisions))
            
            # Count decisions by model
            model_counts = {}
            for decision in filtered_decisions:
                model = decision.get('model_name', 'Unknown')
                model_counts[model] = model_counts.get(model, 0) + 1
            
            # Calculate average confidence (if available)
            confidences = [d.get('confidence') for d in filtered_decisions 
                          if d.get('confidence') is not None]
            avg_confidence = sum(confidences) / len(confidences) if confidences else None
            
            report = {
                'report_timestamp': datetime.now().isoformat(),
                'period': {
                    'start_date': start_date,
                    'end_date': end_date
                },
                'filters': {
                    'model_name': model_name
                },
                'summary': {
                    'total_decisions': total_decisions,
                    'unique_models': unique_models,
                    'unique_cases': unique_cases,
                    'average_confidence': avg_confidence
                },
                'decisions_by_model': model_counts,
                'detailed_decisions': filtered_decisions
            }
            
            return report
            
        except Exception as e:
            self._logger.error(f"Error generating audit report: {e}")
            return {
                'error': str(e),
                'report_timestamp': datetime.now().isoformat()
            }
    
    def list_all_decisions(self) -> List[Dict[str, Any]]:
        """
        List all decision records.
        
        Returns:
            List of all decision records
        """
        try:
            response = self.blockchain.query_chaincode("getAllDecisions", [])
            
            if response.success and response.data:
                return response.data
            else:
                return []
                
        except Exception as e:
            self._logger.error(f"Error listing all decisions: {e}")
            return []
    
    def log_decision_review(self, case_id: str, reviewer: str, 
                          review_result: str, comments: str) -> bool:
        """
        Log a human review of an AI decision.
        
        Args:
            case_id: ID of the case being reviewed
            reviewer: Name/ID of the reviewer
            review_result: Result of the review (approved, rejected, modified)
            comments: Review comments
            
        Returns:
            True if successful, False otherwise
        """
        try:
            review_record = {
                'case_id': case_id,
                'reviewer': reviewer,
                'review_result': review_result,
                'comments': comments,
                'review_timestamp': datetime.now().isoformat()
            }
            
            # Store as metadata update to the original decision
            original_decision = self.get_decision(case_id)
            if original_decision:
                reviews = original_decision.get('metadata', {}).get('reviews', [])
                reviews.append(review_record)
                
                updated_metadata = original_decision.get('metadata', {})
                updated_metadata['reviews'] = reviews
                updated_metadata['last_review'] = review_record
                
                # Update the decision record
                original_decision['metadata'] = updated_metadata
                decision_hash = self._calculate_decision_hash(original_decision)
                
                response = self.blockchain.invoke_chaincode(
                    "logDecision",
                    [case_id, decision_hash, json.dumps(original_decision)]
                )
                
                if response.success:
                    self._logger.info(f"Successfully logged review for case {case_id}")
                    return True
                else:
                    self._logger.error(f"Failed to log review: {response.data}")
                    return False
            else:
                self._logger.error(f"Original decision for case {case_id} not found")
                return False
                
        except Exception as e:
            self._logger.error(f"Error logging decision review: {e}")
            return False
    
    def _calculate_data_hash(self, data: Dict[str, Any]) -> str:
        """
        Calculate SHA-256 hash of data dictionary.
        
        Args:
            data: Data dictionary to hash
            
        Returns:
            Hexadecimal hash string
        """
        data_json = json.dumps(data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(data_json.encode()).hexdigest()
    
    def _calculate_decision_hash(self, decision_record: Dict[str, Any]) -> str:
        """
        Calculate hash for a decision record.
        
        Args:
            decision_record: Decision record to hash
            
        Returns:
            Hexadecimal hash string
        """
        # Create a copy excluding any existing hash
        hashable_record = {k: v for k, v in decision_record.items() 
                          if k != 'decision_hash'}
        
        record_json = json.dumps(hashable_record, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(record_json.encode()).hexdigest()