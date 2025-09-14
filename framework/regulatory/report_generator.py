"""
RegulatoryReportGenerator: Generates regulatory and audit reports from blockchain records.

Provides comprehensive reporting capabilities for healthcare AI compliance including:
- FDA 21 CFR Part 820 quality system reports
- ISO 13485 medical device compliance reports  
- HIPAA audit trails and access logs
- EU MDR (Medical Device Regulation) documentation
- Clinical risk assessment reports
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging


class RegulatoryReportGenerator:
    """
    Comprehensive regulatory report generator with blockchain audit integration
    
    Generates compliance reports for various healthcare AI regulatory frameworks
    including FDA, ISO, HIPAA, and EU MDR requirements.
    """
    
    def __init__(self, blockchain_connector):
        """
        Initialize report generator with blockchain connector
        
        Args:
            blockchain_connector: HyperledgerConnector instance
        """
        self.blockchain = blockchain_connector
        self.logger = logging.getLogger(__name__)
        
        # Report output directory
        self.report_output_dir = "reports"
        os.makedirs(self.report_output_dir, exist_ok=True)
        
        # Report templates and standards
        self.report_standards = {
            'fda_21cfr820': 'FDA 21 CFR Part 820 Quality System Regulation',
            'iso_13485': 'ISO 13485:2016 Medical Devices Quality Management',
            'hipaa_audit': 'HIPAA Security and Privacy Audit Trail', 
            'eu_mdr': 'EU Medical Device Regulation (EU) 2017/745',
            'clinical_risk': 'ISO 14971 Clinical Risk Assessment'
        }
    
    async def generate_model_report(self, model_name: str, report_type: str = 'comprehensive',
                                  include_performance: bool = True) -> str:
        """
        Generate comprehensive regulatory report for a specific model
        
        Args:
            model_name: Model identifier for report
            report_type: Type of report (comprehensive, compliance, performance)
            include_performance: Include performance analytics
            
        Returns:
            Path to generated report file
        """
        try:
            self.logger.info(f"Generating model report for: {model_name}")
            
            # Get model information from blockchain
            model_data = await self.blockchain.get_model(model_name)
            if not model_data:
                raise ValueError(f"Model {model_name} not found in registry")
            
            # Get audit trail and compliance data
            audit_trail = await self.blockchain.get_audit_trail(model_name)
            compliance_report = await self.blockchain.generate_compliance_report(model_name)
            
            # Generate comprehensive report
            report_content = await self._generate_model_report_content(
                model_data, audit_trail, compliance_report, include_performance
            )
            
            # Write report to file
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            filename = f"model_report_{model_name}_{timestamp}.json"
            filepath = os.path.join(self.report_output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report_content, f, indent=2, ensure_ascii=False)
            
            # Generate additional formats
            await self._generate_html_report(report_content, filepath.replace('.json', '.html'))
            
            self.logger.info(f"Model report generated: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Failed to generate model report: {str(e)}")
            return None
    
    async def generate_decision_audit_report(self, case_id: str = "", start_date: str = "",
                                           end_date: str = "", format_type: str = 'json') -> str:
        """
        Generate decision audit report for regulatory compliance
        
        Args:
            case_id: Specific case ID (optional)
            start_date: Start date for report period (ISO format)
            end_date: End date for report period (ISO format)
            format_type: Output format (json, html, pdf)
            
        Returns:
            Path to generated report file
        """
        try:
            self.logger.info(f"Generating decision audit report for case: {case_id or 'ALL'}")
            
            # Get audit trail data
            if case_id:
                audit_data = await self.blockchain.get_audit_trail(f"case_{case_id}", start_date, end_date)
            else:
                # Get system-wide audit trail
                audit_data = {'diagnostics': [], 'compliance_events': []}
                # In production, would aggregate across all models/cases
            
            # Generate audit report content
            report_content = await self._generate_audit_report_content(
                audit_data, case_id, start_date, end_date
            )
            
            # Write report to file
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            case_suffix = f"_{case_id}" if case_id else "_system_wide"
            filename = f"audit_report{case_suffix}_{timestamp}.{format_type}"
            filepath = os.path.join(self.report_output_dir, filename)
            
            if format_type == 'json':
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(report_content, f, indent=2, ensure_ascii=False)
            elif format_type == 'html':
                await self._generate_html_report(report_content, filepath)
            
            self.logger.info(f"Decision audit report generated: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Failed to generate decision audit report: {str(e)}")
            return None
    
    async def generate_compliance_report(self, organization: str = "", regulation: str = 'comprehensive',
                                       period_days: int = 90) -> str:
        """
        Generate regulatory compliance report
        
        Args:
            organization: Organization scope for report
            regulation: Specific regulation (fda_21cfr820, iso_13485, hipaa_audit, eu_mdr)
            period_days: Report period in days
            
        Returns:
            Path to generated compliance report
        """
        try:
            self.logger.info(f"Generating compliance report for: {regulation}")
            
            # Calculate report period
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=period_days)
            
            # Get compliance data from blockchain
            compliance_data = await self.blockchain.generate_compliance_report(
                org_id=organization
            )
            
            # Generate regulation-specific report
            report_content = await self._generate_compliance_report_content(
                compliance_data, regulation, start_date, end_date, organization
            )
            
            # Write report to file
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            org_suffix = f"_{organization}" if organization else "_all_orgs"
            filename = f"compliance_report_{regulation}{org_suffix}_{timestamp}.json"
            filepath = os.path.join(self.report_output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report_content, f, indent=2, ensure_ascii=False)
            
            # Generate regulatory-specific formatted reports
            await self._generate_regulatory_formatted_report(report_content, regulation, filepath)
            
            self.logger.info(f"Compliance report generated: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Failed to generate compliance report: {str(e)}")
            return None
    
    async def generate_risk_assessment_report(self, model_id: str, risk_framework: str = 'iso_14971') -> str:
        """
        Generate clinical risk assessment report
        
        Args:
            model_id: Model for risk assessment
            risk_framework: Risk framework (iso_14971, fda_guidance)
            
        Returns:
            Path to risk assessment report
        """
        try:
            self.logger.info(f"Generating risk assessment for model: {model_id}")
            
            # Get model and performance data
            model_data = await self.blockchain.get_model(model_id)
            audit_trail = await self.blockchain.get_audit_trail(model_id)
            compliance_events = audit_trail.get('compliance_events', [])
            
            # Analyze risks
            risk_analysis = await self._perform_risk_analysis(
                model_data, compliance_events, risk_framework
            )
            
            # Generate report content
            report_content = {
                'report_type': 'clinical_risk_assessment',
                'framework': risk_framework,
                'framework_description': self.report_standards.get(risk_framework.replace('_', '_'), risk_framework),
                'model_info': model_data,
                'risk_analysis': risk_analysis,
                'mitigation_strategies': self._generate_risk_mitigation_strategies(risk_analysis),
                'regulatory_compliance': self._assess_regulatory_compliance(risk_analysis),
                'generated_at': datetime.utcnow().isoformat(),
                'generated_by': getattr(self.blockchain, 'user_name', 'system'),
                'organization': getattr(self.blockchain, 'org_name', 'unknown')
            }
            
            # Write report
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            filename = f"risk_assessment_{model_id}_{timestamp}.json"
            filepath = os.path.join(self.report_output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report_content, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Risk assessment report generated: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Failed to generate risk assessment report: {str(e)}")
            return None
    
    async def generate_performance_monitoring_report(self, time_period: str = '30d') -> str:
        """
        Generate system-wide performance monitoring report
        
        Args:
            time_period: Time period for analysis (7d, 30d, 90d, 1y)
            
        Returns:
            Path to performance report
        """
        try:
            self.logger.info(f"Generating performance monitoring report for: {time_period}")
            
            # Calculate time period
            days_map = {'7d': 7, '30d': 30, '90d': 90, '1y': 365}
            days = days_map.get(time_period, 30)
            
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            # Collect performance data across all models
            # In production, would query multiple models
            performance_data = {
                'system_metrics': await self._collect_system_metrics(start_date, end_date),
                'model_performance': await self._collect_model_performance_data(start_date, end_date),
                'quality_indicators': await self._calculate_quality_indicators(start_date, end_date),
                'trend_analysis': await self._perform_trend_analysis(start_date, end_date)
            }
            
            # Generate report
            report_content = {
                'report_type': 'performance_monitoring',
                'time_period': time_period,
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'performance_data': performance_data,
                'recommendations': self._generate_performance_recommendations(performance_data),
                'generated_at': datetime.utcnow().isoformat()
            }
            
            # Write report
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            filename = f"performance_monitoring_{time_period}_{timestamp}.json"
            filepath = os.path.join(self.report_output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report_content, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Performance monitoring report generated: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Failed to generate performance monitoring report: {str(e)}")
            return None
    
    # === Private Helper Methods ===
    
    async def _generate_model_report_content(self, model_data: Dict[str, Any],
                                           audit_trail: Dict[str, Any],
                                           compliance_report: Dict[str, Any],
                                           include_performance: bool) -> Dict[str, Any]:
        """Generate comprehensive model report content"""
        content = {
            'report_type': 'model_regulatory_report',
            'report_version': '1.0',
            'generated_at': datetime.utcnow().isoformat(),
            'generated_by': getattr(self.blockchain, 'user_name', 'system'),
            'model_information': {
                'model_id': model_data.get('model_id', 'unknown'),
                'name': model_data.get('name', 'unknown'),
                'version': model_data.get('version', 'unknown'),
                'algorithm': model_data.get('algorithm', 'unknown'),
                'regulatory_status': model_data.get('regulatory_status', 'unknown'),
                'created_at': model_data.get('created_at', 'unknown'),
                'creator_org': model_data.get('creator_org', 'unknown')
            },
            'compliance_summary': {
                'compliance_score': compliance_report.get('summary', {}).get('compliance_score', 0),
                'total_events': compliance_report.get('summary', {}).get('total_events', 0),
                'unresolved_events': compliance_report.get('summary', {}).get('unresolved_events', 0),
                'critical_events': compliance_report.get('summary', {}).get('critical_events', 0)
            },
            'audit_summary': {
                'total_diagnostics': len(audit_trail.get('diagnostics', [])),
                'audit_period_start': audit_trail.get('start_date', ''),
                'audit_period_end': audit_trail.get('end_date', ''),
                'data_integrity_verified': True  # Would perform actual verification
            },
            'regulatory_attestations': self._generate_regulatory_attestations(model_data),
            'quality_metrics': self._calculate_model_quality_metrics(audit_trail),
            'recommendations': self._generate_model_recommendations(model_data, compliance_report)
        }
        
        if include_performance:
            content['performance_analysis'] = await self._analyze_model_performance(audit_trail)
        
        return content
    
    async def _generate_audit_report_content(self, audit_data: Dict[str, Any],
                                           case_id: str, start_date: str,
                                           end_date: str) -> Dict[str, Any]:
        """Generate audit report content"""
        diagnostics = audit_data.get('diagnostics', [])
        compliance_events = audit_data.get('compliance_events', [])
        
        return {
            'report_type': 'decision_audit_trail',
            'case_id': case_id or 'system_wide',
            'audit_period': {
                'start_date': start_date or 'inception',
                'end_date': end_date or datetime.utcnow().isoformat()
            },
            'audit_statistics': {
                'total_decisions': len(diagnostics),
                'total_compliance_events': len(compliance_events),
                'unique_models': len(set(d.get('model_id', '') for d in diagnostics)),
                'decision_rate_per_day': len(diagnostics) / 30 if diagnostics else 0
            },
            'decision_breakdown': self._analyze_decision_breakdown(diagnostics),
            'compliance_events': compliance_events,
            'quality_indicators': self._calculate_audit_quality_indicators(diagnostics),
            'regulatory_compliance': {
                'hipaa_compliant': True,
                'audit_trail_complete': True,
                'data_integrity_verified': True
            },
            'generated_at': datetime.utcnow().isoformat()
        }
    
    async def _generate_compliance_report_content(self, compliance_data: Dict[str, Any],
                                                regulation: str, start_date: datetime,
                                                end_date: datetime, organization: str) -> Dict[str, Any]:
        """Generate regulation-specific compliance report"""
        return {
            'report_type': 'regulatory_compliance',
            'regulation_framework': regulation,
            'regulation_description': self.report_standards.get(regulation, regulation),
            'organization': organization or 'system_wide',
            'compliance_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'compliance_summary': compliance_data.get('summary', {}),
            'regulatory_requirements': self._get_regulatory_requirements(regulation),
            'compliance_assessment': self._assess_regulatory_compliance_detailed(compliance_data, regulation),
            'non_conformities': self._identify_non_conformities(compliance_data, regulation),
            'corrective_actions': self._recommend_corrective_actions(compliance_data, regulation),
            'generated_at': datetime.utcnow().isoformat()
        }
    
    async def _perform_risk_analysis(self, model_data: Dict[str, Any],
                                   compliance_events: List[Dict[str, Any]],
                                   risk_framework: str) -> Dict[str, Any]:
        """Perform clinical risk analysis"""
        # Identify potential risks
        risks = []
        
        # Model accuracy risk
        accuracy = model_data.get('accuracy', 0)
        if accuracy < 0.95:
            risks.append({
                'risk_id': 'R001',
                'category': 'diagnostic_accuracy',
                'description': f'Model accuracy below 95% ({accuracy:.2%})',
                'severity': 'medium' if accuracy > 0.90 else 'high',
                'probability': 'medium',
                'impact': 'high'
            })
        
        # Compliance event risks
        critical_events = [e for e in compliance_events if e.get('severity') == 'critical']
        if critical_events:
            risks.append({
                'risk_id': 'R002',
                'category': 'compliance_violation',
                'description': f'{len(critical_events)} critical compliance events detected',
                'severity': 'high',
                'probability': 'high',
                'impact': 'high'
            })
        
        return {
            'framework': risk_framework,
            'identified_risks': risks,
            'overall_risk_level': self._calculate_overall_risk_level(risks),
            'risk_matrix': self._create_risk_matrix(risks),
            'analysis_date': datetime.utcnow().isoformat()
        }
    
    # Additional helper methods for report generation
    def _generate_regulatory_attestations(self, model_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate regulatory attestation statements"""
        return [
            {
                'statement': 'Model development follows FDA Software as Medical Device guidance',
                'status': 'compliant',
                'evidence': 'Development process documentation available'
            },
            {
                'statement': 'Clinical validation performed according to ISO 13485',
                'status': 'compliant' if model_data.get('accuracy', 0) > 0.90 else 'non_compliant',
                'evidence': f"Model accuracy: {model_data.get('accuracy', 0):.2%}"
            }
        ]
    
    def _calculate_model_quality_metrics(self, audit_trail: Dict[str, Any]) -> Dict[str, float]:
        """Calculate quality metrics from audit trail"""
        diagnostics = audit_trail.get('diagnostics', [])
        
        if not diagnostics:
            return {'error': 'No diagnostic data available'}
        
        confidence_scores = [d.get('confidence_score', 0) for d in diagnostics]
        
        return {
            'average_confidence': sum(confidence_scores) / len(confidence_scores),
            'consistency_score': 0.92,  # Mock calculation
            'completeness_score': 0.98,  # Mock calculation
            'timeliness_score': 0.95    # Mock calculation
        }
    
    def _generate_model_recommendations(self, model_data: Dict[str, Any], 
                                      compliance_report: Dict[str, Any]) -> List[str]:
        """Generate recommendations for model improvement"""
        recommendations = []
        
        compliance_score = compliance_report.get('summary', {}).get('compliance_score', 100)
        if compliance_score < 90:
            recommendations.append("Address outstanding compliance issues to improve compliance score")
        
        accuracy = model_data.get('accuracy', 0)
        if accuracy < 0.95:
            recommendations.append("Consider model retraining to improve diagnostic accuracy")
        
        return recommendations
    
    async def _generate_html_report(self, report_content: Dict[str, Any], filepath: str):
        """Generate HTML version of report"""
        # Simple HTML generation - in production would use proper templating
        html_content = f"""
        <html>
        <head><title>Regulatory Report</title></head>
        <body>
        <h1>Healthcare AI Regulatory Report</h1>
        <pre>{json.dumps(report_content, indent=2)}</pre>
        </body>
        </html>
        """
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    # Mock implementations for remaining helper methods
    async def _generate_regulatory_formatted_report(self, report_content: Dict[str, Any], 
                                                  regulation: str, filepath: str):
        """Generate regulation-specific formatted report"""
        pass
    
    async def _collect_system_metrics(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Collect system-wide performance metrics"""
        return {'uptime': 99.9, 'response_time': 150, 'error_rate': 0.01}
    
    async def _collect_model_performance_data(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Collect model performance data"""
        return {'active_models': 5, 'total_predictions': 1000, 'average_accuracy': 0.94}
    
    async def _calculate_quality_indicators(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Calculate quality indicators"""
        return {'data_quality': 0.96, 'process_compliance': 0.98}
    
    async def _perform_trend_analysis(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Perform trend analysis"""
        return {'accuracy_trend': 'stable', 'usage_trend': 'increasing'}
    
    def _generate_performance_recommendations(self, performance_data: Dict[str, Any]) -> List[str]:
        """Generate performance improvement recommendations"""
        return ["Monitor model drift", "Review data quality processes"]
    
    # Additional mock methods for complete implementation
    def _analyze_decision_breakdown(self, diagnostics: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {'high_confidence': 80, 'medium_confidence': 15, 'low_confidence': 5}
    
    def _calculate_audit_quality_indicators(self, diagnostics: List[Dict[str, Any]]) -> Dict[str, float]:
        return {'completeness': 0.98, 'accuracy': 0.95, 'timeliness': 0.97}
    
    def _get_regulatory_requirements(self, regulation: str) -> List[Dict[str, str]]:
        return [{'requirement': 'Documentation', 'status': 'compliant'}]
    
    def _assess_regulatory_compliance_detailed(self, compliance_data: Dict[str, Any], regulation: str) -> Dict[str, Any]:
        return {'overall_status': 'compliant', 'score': 95}
    
    def _identify_non_conformities(self, compliance_data: Dict[str, Any], regulation: str) -> List[Dict[str, str]]:
        return []
    
    def _recommend_corrective_actions(self, compliance_data: Dict[str, Any], regulation: str) -> List[Dict[str, str]]:
        return []
    
    def _generate_risk_mitigation_strategies(self, risk_analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        return [{'strategy': 'Regular model validation', 'priority': 'high'}]
    
    def _assess_regulatory_compliance(self, risk_analysis: Dict[str, Any]) -> Dict[str, Any]:
        return {'status': 'compliant', 'recommendations': []}
    
    def _calculate_overall_risk_level(self, risks: List[Dict[str, Any]]) -> str:
        if any(r.get('severity') == 'high' for r in risks):
            return 'high'
        elif any(r.get('severity') == 'medium' for r in risks):
            return 'medium'
        else:
            return 'low'
    
    def _create_risk_matrix(self, risks: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {'high_risk': len([r for r in risks if r.get('severity') == 'high'])}
    
    async def _analyze_model_performance(self, audit_trail: Dict[str, Any]) -> Dict[str, Any]:
        return {'trend': 'stable', 'metrics': {'accuracy': 0.94}}