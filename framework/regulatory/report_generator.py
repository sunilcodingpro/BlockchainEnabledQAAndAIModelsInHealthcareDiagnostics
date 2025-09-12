"""
RegulatoryReportGenerator: Generates regulatory and audit reports from blockchain records.
"""
import json
import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime


class RegulatoryReportGenerator:
    """
    Generator for regulatory compliance and audit reports.
    
    This class retrieves data from blockchain records and formats
    comprehensive reports for regulatory compliance, auditing,
    and quality assurance purposes.
    """
    
    def __init__(self, blockchain_connector, output_dir: str = "reports"):
        """
        Initialize the regulatory report generator.
        
        Args:
            blockchain_connector: Instance of HyperledgerConnector for blockchain operations
            output_dir: Directory to save generated reports
        """
        self.blockchain = blockchain_connector
        self.output_dir = output_dir
        self._logger = logging.getLogger(__name__)
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_model_report(self, model_name: str, format_type: str = "json") -> Optional[str]:
        """
        Generate comprehensive report for a specific model.
        
        Args:
            model_name: Name of the model to generate report for
            format_type: Output format ("json", "html", "pdf")
            
        Returns:
            Path to generated report file or None if failed
        """
        try:
            self._logger.info(f"Generating model report for {model_name}")
            
            # Retrieve model information from blockchain
            model_info = self._get_model_info(model_name)
            if not model_info:
                self._logger.error(f"Model {model_name} not found")
                return None
            
            # Get related decisions
            decisions = self._get_model_decisions(model_name)
            
            # Get audit trail
            audit_trail = self._get_model_audit_trail(model_name)
            
            # Compile report data
            report_data = {
                'report_type': 'model_compliance_report',
                'generated_at': datetime.now().isoformat(),
                'model_information': model_info,
                'compliance_status': self._assess_model_compliance(model_info, decisions),
                'usage_statistics': self._calculate_usage_statistics(decisions),
                'decision_history': decisions,
                'audit_trail': audit_trail,
                'regulatory_notes': self._generate_regulatory_notes(model_info, decisions)
            }
            
            # Generate report file
            filename = f"model_report_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            return self._save_report(report_data, filename, format_type)
            
        except Exception as e:
            self._logger.error(f"Error generating model report: {e}")
            return None
    
    def generate_decision_audit_report(self, case_id: str, format_type: str = "json") -> Optional[str]:
        """
        Generate audit report for a specific decision case.
        
        Args:
            case_id: ID of the case to generate report for
            format_type: Output format ("json", "html", "pdf")
            
        Returns:
            Path to generated report file or None if failed
        """
        try:
            self._logger.info(f"Generating decision audit report for case {case_id}")
            
            # Retrieve decision information
            decision_info = self._get_decision_info(case_id)
            if not decision_info:
                self._logger.error(f"Decision for case {case_id} not found")
                return None
            
            # Get model information
            model_name = decision_info.get('model_name')
            model_info = self._get_model_info(model_name) if model_name else {}
            
            # Get data provenance
            input_data_hash = decision_info.get('input_data_hash')
            provenance_info = self._get_provenance_by_hash(input_data_hash) if input_data_hash else {}
            
            # Perform integrity audit
            integrity_audit = self._audit_decision_integrity(case_id, decision_info)
            
            # Compile report data
            report_data = {
                'report_type': 'decision_audit_report',
                'generated_at': datetime.now().isoformat(),
                'case_information': {
                    'case_id': case_id,
                    'decision_timestamp': decision_info.get('timestamp'),
                    'decision': decision_info.get('decision'),
                    'confidence': decision_info.get('confidence'),
                    'model_used': model_name
                },
                'model_information': model_info,
                'data_provenance': provenance_info,
                'decision_details': decision_info,
                'integrity_audit': integrity_audit,
                'explainability': decision_info.get('explanation', {}),
                'compliance_assessment': self._assess_decision_compliance(decision_info),
                'regulatory_notes': self._generate_decision_regulatory_notes(decision_info)
            }
            
            # Generate report file
            filename = f"decision_audit_{case_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            return self._save_report(report_data, filename, format_type)
            
        except Exception as e:
            self._logger.error(f"Error generating decision audit report: {e}")
            return None
    
    def generate_comprehensive_audit_report(self, start_date: Optional[str] = None,
                                          end_date: Optional[str] = None,
                                          model_filter: Optional[str] = None,
                                          format_type: str = "json") -> Optional[str]:
        """
        Generate comprehensive audit report covering all activities.
        
        Args:
            start_date: Start date for report (ISO format)
            end_date: End date for report (ISO format)
            model_filter: Optional model name filter
            format_type: Output format ("json", "html", "pdf")
            
        Returns:
            Path to generated report file or None if failed
        """
        try:
            self._logger.info("Generating comprehensive audit report")
            
            # Get all models
            all_models = self._get_all_models()
            
            # Get all decisions
            all_decisions = self._get_all_decisions()
            
            # Get all provenance records
            all_provenance = self._get_all_provenance()
            
            # Apply filters
            if model_filter:
                all_models = [m for m in all_models if m.get('name') == model_filter]
                all_decisions = [d for d in all_decisions if d.get('model_name') == model_filter]
            
            if start_date:
                all_decisions = [d for d in all_decisions if d.get('timestamp', '') >= start_date]
                all_provenance = [p for p in all_provenance if p.get('timestamp', '') >= start_date]
            
            if end_date:
                all_decisions = [d for d in all_decisions if d.get('timestamp', '') <= end_date]
                all_provenance = [p for p in all_provenance if p.get('timestamp', '') <= end_date]
            
            # Generate statistics
            statistics = self._generate_comprehensive_statistics(all_models, all_decisions, all_provenance)
            
            # Compile report data
            report_data = {
                'report_type': 'comprehensive_audit_report',
                'generated_at': datetime.now().isoformat(),
                'report_period': {
                    'start_date': start_date,
                    'end_date': end_date
                },
                'filters': {
                    'model_filter': model_filter
                },
                'executive_summary': statistics,
                'model_registry': {
                    'total_models': len(all_models),
                    'models': all_models
                },
                'decision_audit': {
                    'total_decisions': len(all_decisions),
                    'decisions': all_decisions
                },
                'data_provenance': {
                    'total_samples': len(all_provenance),
                    'provenance_records': all_provenance
                },
                'compliance_summary': self._generate_compliance_summary(all_models, all_decisions),
                'recommendations': self._generate_recommendations(statistics)
            }
            
            # Generate report file
            filename = f"comprehensive_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            return self._save_report(report_data, filename, format_type)
            
        except Exception as e:
            self._logger.error(f"Error generating comprehensive audit report: {e}")
            return None
    
    def _get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve model information from blockchain."""
        try:
            response = self.blockchain.query_chaincode("getModel", [model_name])
            return response.data if response.success else None
        except Exception as e:
            self._logger.error(f"Error retrieving model info: {e}")
            return None
    
    def _get_decision_info(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve decision information from blockchain."""
        try:
            response = self.blockchain.query_chaincode("getDecision", [case_id])
            return response.data if response.success else None
        except Exception as e:
            self._logger.error(f"Error retrieving decision info: {e}")
            return None
    
    def _get_model_decisions(self, model_name: str) -> List[Dict[str, Any]]:
        """Get all decisions made by a specific model."""
        try:
            all_decisions = self._get_all_decisions()
            return [d for d in all_decisions if d.get('model_name') == model_name]
        except Exception as e:
            self._logger.error(f"Error retrieving model decisions: {e}")
            return []
    
    def _get_all_models(self) -> List[Dict[str, Any]]:
        """Get all registered models."""
        try:
            response = self.blockchain.query_chaincode("getAllModels", [])
            return response.data if response.success else []
        except Exception as e:
            self._logger.error(f"Error retrieving all models: {e}")
            return []
    
    def _get_all_decisions(self) -> List[Dict[str, Any]]:
        """Get all decision records."""
        try:
            response = self.blockchain.query_chaincode("getAllDecisions", [])
            return response.data if response.success else []
        except Exception as e:
            self._logger.error(f"Error retrieving all decisions: {e}")
            return []
    
    def _get_all_provenance(self) -> List[Dict[str, Any]]:
        """Get all provenance records."""
        try:
            response = self.blockchain.query_chaincode("getAllProvenance", [])
            return response.data if response.success else []
        except Exception as e:
            self._logger.error(f"Error retrieving all provenance: {e}")
            return []
    
    def _get_provenance_by_hash(self, data_hash: str) -> Optional[Dict[str, Any]]:
        """Find provenance record by data hash."""
        try:
            all_provenance = self._get_all_provenance()
            for record in all_provenance:
                if record.get('sample_hash') == data_hash:
                    return record
            return None
        except Exception as e:
            self._logger.error(f"Error retrieving provenance by hash: {e}")
            return None
    
    def _get_model_audit_trail(self, model_name: str) -> List[Dict[str, Any]]:
        """Get audit trail for a model from blockchain transaction history."""
        try:
            transaction_history = self.blockchain.get_transaction_history()
            model_transactions = []
            
            for tx in transaction_history:
                if (tx.get('function') == 'registerModel' and 
                    tx.get('args', [{}])[0] == model_name):
                    model_transactions.append({
                        'transaction_id': tx.get('tx_id'),
                        'timestamp': tx.get('timestamp'),
                        'action': 'model_registration',
                        'block_number': tx.get('block_number')
                    })
            
            return model_transactions
        except Exception as e:
            self._logger.error(f"Error retrieving model audit trail: {e}")
            return []
    
    def _assess_model_compliance(self, model_info: Dict[str, Any], 
                               decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess compliance status of a model."""
        compliance = {
            'overall_status': 'COMPLIANT',
            'issues': [],
            'recommendations': []
        }
        
        # Check if model has required metadata
        required_fields = ['name', 'hash', 'timestamp']
        for field in required_fields:
            if field not in model_info:
                compliance['issues'].append(f"Missing required field: {field}")
                compliance['overall_status'] = 'NON_COMPLIANT'
        
        # Check decision trail
        if not decisions:
            compliance['issues'].append("No decisions recorded for this model")
            compliance['recommendations'].append("Ensure all model predictions are logged")
        
        # Check for explainability
        decisions_with_explanations = [d for d in decisions if d.get('explanation')]
        if decisions and len(decisions_with_explanations) / len(decisions) < 0.8:
            compliance['issues'].append("Less than 80% of decisions have explanations")
            compliance['recommendations'].append("Improve explainability coverage")
        
        return compliance
    
    def _assess_decision_compliance(self, decision_info: Dict[str, Any]) -> Dict[str, Any]:
        """Assess compliance of a specific decision."""
        compliance = {
            'overall_status': 'COMPLIANT',
            'issues': [],
            'recommendations': []
        }
        
        # Check required fields
        required_fields = ['case_id', 'model_name', 'decision', 'timestamp']
        for field in required_fields:
            if field not in decision_info:
                compliance['issues'].append(f"Missing required field: {field}")
                compliance['overall_status'] = 'NON_COMPLIANT'
        
        # Check for explanation
        if not decision_info.get('explanation'):
            compliance['issues'].append("No explanation provided")
            compliance['recommendations'].append("Add explainability analysis")
        
        # Check confidence score
        if decision_info.get('confidence') is None:
            compliance['recommendations'].append("Consider adding confidence scores")
        
        return compliance
    
    def _audit_decision_integrity(self, case_id: str, 
                                decision_info: Dict[str, Any]) -> Dict[str, Any]:
        """Audit integrity of a decision record."""
        import hashlib
        
        # Recalculate hash to verify integrity
        stored_hash = decision_info.get('decision_hash')
        
        # Create hashable copy
        hashable_data = {k: v for k, v in decision_info.items() if k != 'decision_hash'}
        calculated_hash = hashlib.sha256(
            json.dumps(hashable_data, sort_keys=True).encode()
        ).hexdigest()
        
        return {
            'case_id': case_id,
            'integrity_status': 'PASSED' if stored_hash == calculated_hash else 'FAILED',
            'stored_hash': stored_hash,
            'calculated_hash': calculated_hash,
            'audit_timestamp': datetime.now().isoformat()
        }
    
    def _calculate_usage_statistics(self, decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate usage statistics for decisions."""
        if not decisions:
            return {'total_decisions': 0}
        
        # Calculate basic stats
        total_decisions = len(decisions)
        
        # Count decisions by date
        decisions_by_date = {}
        for decision in decisions:
            date = decision.get('timestamp', '')[:10]  # Extract date part
            decisions_by_date[date] = decisions_by_date.get(date, 0) + 1
        
        # Calculate confidence stats if available
        confidences = [d.get('confidence') for d in decisions if d.get('confidence') is not None]
        
        return {
            'total_decisions': total_decisions,
            'decisions_by_date': decisions_by_date,
            'average_confidence': sum(confidences) / len(confidences) if confidences else None,
            'decisions_with_confidence': len(confidences),
            'decisions_with_explanations': len([d for d in decisions if d.get('explanation')])
        }
    
    def _generate_comprehensive_statistics(self, models: List[Dict[str, Any]],
                                         decisions: List[Dict[str, Any]],
                                         provenance: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive statistics."""
        return {
            'total_models': len(models),
            'total_decisions': len(decisions),
            'total_data_samples': len(provenance),
            'models_with_decisions': len(set(d.get('model_name') for d in decisions)),
            'average_decisions_per_model': len(decisions) / max(len(models), 1),
            'data_sources': list(set(p.get('provenance', {}).get('source') 
                                   for p in provenance if p.get('provenance', {}).get('source')))
        }
    
    def _generate_compliance_summary(self, models: List[Dict[str, Any]],
                                   decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate overall compliance summary."""
        total_models = len(models)
        compliant_models = 0
        
        for model in models:
            model_decisions = [d for d in decisions if d.get('model_name') == model.get('name')]
            compliance = self._assess_model_compliance(model, model_decisions)
            if compliance['overall_status'] == 'COMPLIANT':
                compliant_models += 1
        
        return {
            'total_models_assessed': total_models,
            'compliant_models': compliant_models,
            'compliance_rate': compliant_models / max(total_models, 1) * 100,
            'non_compliant_models': total_models - compliant_models
        }
    
    def _generate_recommendations(self, statistics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on statistics."""
        recommendations = []
        
        if statistics.get('total_models', 0) == 0:
            recommendations.append("No models registered. Consider registering AI/ML models for tracking.")
        
        if statistics.get('total_decisions', 0) == 0:
            recommendations.append("No decisions logged. Ensure all AI predictions are being recorded.")
        
        models_with_decisions = statistics.get('models_with_decisions', 0)
        total_models = statistics.get('total_models', 0)
        
        if total_models > 0 and models_with_decisions / total_models < 0.5:
            recommendations.append("Less than 50% of models have recorded decisions. Review model usage tracking.")
        
        return recommendations
    
    def _generate_regulatory_notes(self, model_info: Dict[str, Any], 
                                 decisions: List[Dict[str, Any]]) -> List[str]:
        """Generate regulatory compliance notes."""
        notes = []
        
        # FDA/CE marking notes
        notes.append("Ensure model validation meets FDA Software as Medical Device (SaMD) guidelines")
        notes.append("Maintain audit trail for CE marking compliance under MDR")
        
        # GDPR/Privacy notes
        if any('patient_id' in str(d) or 'personal' in str(d) for d in decisions):
            notes.append("Personal data detected - ensure GDPR compliance for data processing")
        
        # Quality notes
        notes.append("Regular model performance monitoring recommended")
        notes.append("Maintain explainability records for clinical decision support")
        
        return notes
    
    def _generate_decision_regulatory_notes(self, decision_info: Dict[str, Any]) -> List[str]:
        """Generate regulatory notes for a specific decision."""
        notes = []
        
        notes.append("Decision audit trail maintained for regulatory compliance")
        
        if decision_info.get('explanation'):
            notes.append("Explainability information available for clinical review")
        else:
            notes.append("Consider adding explainability analysis for better compliance")
        
        if decision_info.get('confidence'):
            notes.append(f"Confidence score: {decision_info.get('confidence'):.2f}")
        
        return notes
    
    def _save_report(self, report_data: Dict[str, Any], filename: str, 
                    format_type: str) -> str:
        """Save report to file."""
        try:
            if format_type.lower() == "json":
                filepath = os.path.join(self.output_dir, f"{filename}.json")
                with open(filepath, 'w') as f:
                    json.dump(report_data, f, indent=2, default=str)
            
            elif format_type.lower() == "html":
                filepath = os.path.join(self.output_dir, f"{filename}.html")
                html_content = self._generate_html_report(report_data)
                with open(filepath, 'w') as f:
                    f.write(html_content)
            
            else:  # Default to JSON
                filepath = os.path.join(self.output_dir, f"{filename}.json")
                with open(filepath, 'w') as f:
                    json.dump(report_data, f, indent=2, default=str)
            
            self._logger.info(f"Report saved to: {filepath}")
            return filepath
            
        except Exception as e:
            self._logger.error(f"Error saving report: {e}")
            return None
    
    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """Generate HTML version of report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Regulatory Report - {report_data.get('report_type', 'Unknown')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; }}
                .section {{ margin: 20px 0; }}
                .compliance-pass {{ color: green; }}
                .compliance-fail {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Regulatory Compliance Report</h1>
                <p>Report Type: {report_data.get('report_type', 'Unknown')}</p>
                <p>Generated: {report_data.get('generated_at', 'Unknown')}</p>
            </div>
            
            <div class="section">
                <h2>Summary</h2>
                <pre>{json.dumps(report_data, indent=2, default=str)}</pre>
            </div>
        </body>
        </html>
        """
        return html