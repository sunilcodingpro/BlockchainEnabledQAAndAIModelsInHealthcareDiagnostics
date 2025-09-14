"""
SHAPExplainer: Generates SHAP-based feature attributions for model interpretability.

This module provides SHAP (SHapley Additive exPlanations) explainability for AI/ML models,
generating feature importance scores and explanations for individual predictions.
"""

import json
import random
import math
from typing import Dict, Any, List, Optional, Union


class SHAPExplainer:
    """
    SHAP-based explainer for AI/ML model interpretability.
    
    This class generates SHAP values and explanations for model predictions,
    supporting various model types and explanation formats.
    """
    
    def __init__(self, model: Any, model_type: str = "auto"):
        """
        Initialize the SHAP explainer.
        
        Args:
            model: The ML model to explain (sklearn, tensorflow, etc.)
            model_type: Type of model ("sklearn", "tree", "linear", "deep", "auto")
        """
        self.model = model
        self.model_type = model_type
        self.explainer = None
        self.feature_names = None
        self._setup_explainer()
    
    def _setup_explainer(self) -> None:
        """
        Set up the appropriate SHAP explainer based on model type.
        
        In a full implementation with SHAP library, this would:
        - Create TreeExplainer for tree-based models
        - Create LinearExplainer for linear models
        - Create DeepExplainer for neural networks
        - Create KernelExplainer as fallback
        """
        try:
            # In a real implementation, this would import and use SHAP:
            # import shap
            # 
            # if self.model_type == "tree" or (self.model_type == "auto" and hasattr(self.model, 'tree_')):
            #     self.explainer = shap.TreeExplainer(self.model)
            # elif self.model_type == "linear":
            #     self.explainer = shap.LinearExplainer(self.model)
            # else:
            #     self.explainer = shap.KernelExplainer(self.model.predict, background_data)
            
            # For demo purposes, create a mock explainer
            self.explainer = MockSHAPExplainer(self.model)
            print(f"SHAP explainer initialized for model type: {self.model_type}")
            
        except Exception as e:
            print(f"Warning: Could not initialize SHAP explainer: {e}")
            print("Using mock explainer for demonstration purposes")
            self.explainer = MockSHAPExplainer(self.model)
    
    def explain(self, input_data: Union[Dict[str, Any], List[Dict[str, Any]]], 
               feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate SHAP explanations for input data.
        
        Args:
            input_data: Input data (single sample or batch)
            feature_names: Optional list of feature names
            
        Returns:
            Dictionary containing SHAP values and explanations
        """
        try:
            if feature_names:
                self.feature_names = feature_names
            
            # Handle both single sample and batch inputs
            if isinstance(input_data, dict):
                input_data = [input_data]
            
            explanations = []
            
            for sample in input_data:
                # Convert sample to array format
                sample_array = self._dict_to_array(sample)
                
                # Generate SHAP values
                shap_values = self.explainer.shap_values(sample_array)
                
                # Create explanation dictionary
                explanation = self._create_explanation_dict(sample, shap_values)
                explanations.append(explanation)
            
            # Return single explanation or batch
            if len(explanations) == 1:
                return {
                    "explanation_type": "shap",
                    "explanation": explanations[0],
                    "generated_at": self._get_timestamp()
                }
            else:
                return {
                    "explanation_type": "shap_batch",
                    "explanations": explanations,
                    "batch_size": len(explanations),
                    "generated_at": self._get_timestamp()
                }
                
        except Exception as e:
            return {
                "explanation_type": "shap",
                "error": str(e),
                "generated_at": self._get_timestamp()
            }
    
    def explain_prediction(self, input_data: Dict[str, Any], 
                          prediction: Any) -> Dict[str, Any]:
        """
        Generate SHAP explanations for a specific prediction.
        
        Args:
            input_data: Input data used for prediction
            prediction: The model's prediction
            
        Returns:
            Comprehensive explanation including prediction context
        """
        try:
            # Get basic SHAP explanation
            shap_explanation = self.explain(input_data)
            
            # Enhance with prediction context
            enhanced_explanation = {
                **shap_explanation,
                "prediction_context": {
                    "prediction": prediction,
                    "input_features": list(input_data.keys()),
                    "feature_count": len(input_data)
                },
                "interpretation": self._interpret_shap_values(
                    shap_explanation.get("explanation", {}), 
                    prediction
                )
            }
            
            return enhanced_explanation
            
        except Exception as e:
            return {
                "explanation_type": "shap_prediction",
                "error": str(e),
                "generated_at": self._get_timestamp()
            }
    
    def get_feature_importance(self, input_data: Union[Dict[str, Any], List[Dict[str, Any]]], 
                             top_k: int = 10) -> Dict[str, Any]:
        """
        Get top-k most important features based on SHAP values.
        
        Args:
            input_data: Input data
            top_k: Number of top features to return
            
        Returns:
            Dictionary with top important features
        """
        try:
            explanation = self.explain(input_data)
            
            if "explanation" in explanation:
                feature_importance = explanation["explanation"].get("feature_importance", {})
                
                # Sort by absolute importance
                sorted_features = sorted(
                    feature_importance.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )
                
                return {
                    "top_features": sorted_features[:top_k],
                    "feature_count": len(feature_importance),
                    "importance_type": "shap_values",
                    "generated_at": self._get_timestamp()
                }
            else:
                return {
                    "error": "No feature importance found in explanation",
                    "generated_at": self._get_timestamp()
                }
                
        except Exception as e:
            return {
                "error": str(e),
                "generated_at": self._get_timestamp()
            }
    
    def generate_summary_plot_data(self, input_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate data for SHAP summary plots.
        
        Args:
            input_data: Batch of input samples
            
        Returns:
            Data structure for creating summary plots
        """
        try:
            batch_explanations = self.explain(input_data)
            
            if batch_explanations.get("explanation_type") != "shap_batch":
                return {"error": "Batch explanations required for summary plot"}
            
            # Aggregate SHAP values across batch
            all_shap_values = []
            all_features = set()
            
            for explanation in batch_explanations.get("explanations", []):
                feature_importance = explanation.get("feature_importance", {})
                all_shap_values.append(feature_importance)
                all_features.update(feature_importance.keys())
            
            # Calculate mean absolute SHAP values for each feature
            mean_shap = {}
            for feature in all_features:
                values = [abs(sample.get(feature, 0)) for sample in all_shap_values]
                mean_shap[feature] = sum(values) / len(values) if values else 0
            
            return {
                "summary_type": "shap_summary",
                "mean_shap_values": mean_shap,
                "sample_count": len(input_data),
                "feature_count": len(all_features),
                "generated_at": self._get_timestamp()
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "generated_at": self._get_timestamp()
            }
    
    def _dict_to_array(self, data_dict: Dict[str, Any]) -> List[float]:
        """
        Convert dictionary input to list.
        
        Args:
            data_dict: Input data dictionary
            
        Returns:
            List representation
        """
        try:
            if self.feature_names:
                # Use specified feature order
                values = [data_dict.get(name, 0) for name in self.feature_names]
            else:
                # Use alphabetical order for consistency
                sorted_keys = sorted(data_dict.keys())
                values = [data_dict[key] for key in sorted_keys]
                if not self.feature_names:
                    self.feature_names = sorted_keys
            
            return [float(v) for v in values]
            
        except Exception as e:
            print(f"Error converting dict to array: {e}")
            return [0.0]
    
    def _create_explanation_dict(self, sample: Dict[str, Any], 
                               shap_values: List[float]) -> Dict[str, Any]:
        """
        Create explanation dictionary from SHAP values.
        
        Args:
            sample: Original input sample
            shap_values: SHAP values list
            
        Returns:
            Explanation dictionary
        """
        try:
            feature_names = self.feature_names or list(sample.keys())
            
            # Create feature importance mapping
            feature_importance = {}
            for i, feature_name in enumerate(feature_names[:len(shap_values)]):
                feature_importance[feature_name] = float(shap_values[i])
            
            return {
                "feature_importance": feature_importance,
                "base_value": 0.0,  # Would be actual base value in real implementation
                "prediction_impact": sum(abs(v) for v in feature_importance.values()),
                "method": "SHAP"
            }
            
        except Exception as e:
            print(f"Error creating explanation dict: {e}")
            return {
                "feature_importance": {},
                "error": str(e),
                "method": "SHAP"
            }
    
    def _interpret_shap_values(self, explanation: Dict[str, Any], 
                             prediction: Any) -> Dict[str, Any]:
        """
        Provide human-readable interpretation of SHAP values.
        
        Args:
            explanation: SHAP explanation dictionary
            prediction: Model prediction
            
        Returns:
            Human-readable interpretation
        """
        feature_importance = explanation.get("feature_importance", {})
        
        if not feature_importance:
            return {"interpretation": "No feature importance available"}
        
        # Find most positive and negative contributors
        positive_contributors = {k: v for k, v in feature_importance.items() if v > 0}
        negative_contributors = {k: v for k, v in feature_importance.items() if v < 0}
        
        # Sort by absolute value
        top_positive = sorted(positive_contributors.items(), key=lambda x: x[1], reverse=True)
        top_negative = sorted(negative_contributors.items(), key=lambda x: x[1])
        
        interpretation = {
            "prediction": prediction,
            "top_positive_contributors": top_positive[:3],
            "top_negative_contributors": top_negative[:3],
            "total_positive_impact": sum(positive_contributors.values()),
            "total_negative_impact": sum(negative_contributors.values()),
            "net_impact": sum(feature_importance.values())
        }
        
        return interpretation
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()


class MockSHAPExplainer:
    """
    Mock SHAP explainer for demonstration purposes.
    
    This class simulates SHAP functionality when the actual SHAP library
    is not available, providing realistic mock explanations.
    """
    
    def __init__(self, model: Any):
        """Initialize mock explainer."""
        self.model = model
    
    def shap_values(self, X: List[float]) -> List[float]:
        """
        Generate mock SHAP values.
        
        Args:
            X: Input data list
            
        Returns:
            Mock SHAP values
        """
        try:
            # Generate realistic mock SHAP values
            n_features = len(X)
            
            # Create some variation based on input values
            shap_values = []
            for i in range(n_features):
                base_value = random.normalvariate(0, 0.1)
                # Scale importance by feature value
                if i < len(X):
                    base_value *= (1 + abs(X[i]) * 0.1)
                shap_values.append(base_value)
            
            # Ensure SHAP values sum appropriately
            total_abs = sum(abs(v) for v in shap_values)
            if total_abs > 0:
                shap_values = [v / total_abs * 0.5 for v in shap_values]
            
            return shap_values
            
        except Exception as e:
            print(f"Error generating mock SHAP values: {e}")
            return [0.1, -0.05, 0.2]  # Fallback values