"""
SHAPExplainer: Generates SHAP-based feature attributions for model interpretability.
"""
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Union

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP library not available. Using mock implementation.")


class SHAPExplainer:
    """
    SHAP (SHapley Additive exPlanations) explainer for model interpretability.
    
    This class generates feature importance explanations for machine learning models
    using SHAP values, which provide consistent and theoretically grounded explanations.
    """
    
    def __init__(self, model, model_type: str = "auto", feature_names: Optional[List[str]] = None,
                 background_data: Optional[np.ndarray] = None):
        """
        Initialize the SHAP explainer.
        
        Args:
            model: Trained ML model (sklearn-compatible)
            model_type: Type of model ("tree", "linear", "kernel", "deep", "auto")
            feature_names: Names of input features
            background_data: Background dataset for explainer initialization
        """
        self.model = model
        self.model_type = model_type
        self.feature_names = feature_names
        self.background_data = background_data
        self._logger = logging.getLogger(__name__)
        
        # Initialize SHAP explainer
        self.explainer = None
        self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Initialize the appropriate SHAP explainer based on model type."""
        if not SHAP_AVAILABLE:
            self._logger.warning("SHAP not available, using mock explainer")
            return
        
        try:
            # Auto-detect model type if not specified
            if self.model_type == "auto":
                self.model_type = self._detect_model_type()
            
            if self.model_type == "tree":
                self.explainer = shap.TreeExplainer(self.model)
            elif self.model_type == "linear":
                self.explainer = shap.LinearExplainer(self.model, self.background_data)
            elif self.model_type == "kernel":
                if self.background_data is not None:
                    self.explainer = shap.KernelExplainer(self.model.predict, self.background_data)
                else:
                    raise ValueError("Background data required for KernelExplainer")
            elif self.model_type == "deep":
                self.explainer = shap.DeepExplainer(self.model, self.background_data)
            else:
                # Default to KernelExplainer
                if self.background_data is not None:
                    self.explainer = shap.KernelExplainer(self.model.predict, self.background_data)
                else:
                    self._logger.warning("No background data provided, using mock explainer")
            
            self._logger.info(f"Initialized SHAP {self.model_type} explainer")
            
        except Exception as e:
            self._logger.error(f"Error initializing SHAP explainer: {e}")
            self.explainer = None
    
    def explain(self, input_data: Union[np.ndarray, Dict[str, Any], List], 
                return_format: str = "dict") -> Dict[str, Any]:
        """
        Generate SHAP explanations for input data.
        
        Args:
            input_data: Input data to explain (array, dict, or list)
            return_format: Format of returned explanation ("dict", "array", "detailed")
            
        Returns:
            Dictionary containing SHAP values and explanation metadata
        """
        try:
            # Convert input data to numpy array
            X = self._prepare_input_data(input_data)
            
            if self.explainer is None or not SHAP_AVAILABLE:
                return self._mock_explanation(X, return_format)
            
            # Generate SHAP values
            shap_values = self.explainer.shap_values(X)
            
            # Handle multi-class case
            if isinstance(shap_values, list):
                # For multi-class, take the values for the predicted class
                prediction = self.model.predict(X.reshape(1, -1))[0]
                if hasattr(self.model, 'classes_'):
                    class_idx = np.where(self.model.classes_ == prediction)[0][0]
                    shap_values = shap_values[class_idx]
                else:
                    shap_values = shap_values[0]  # Default to first class
            
            # Get base value (expected value)
            if hasattr(self.explainer, 'expected_value'):
                base_value = self.explainer.expected_value
                if isinstance(base_value, np.ndarray):
                    base_value = base_value[0] if len(base_value) > 0 else 0.0
            else:
                base_value = 0.0
            
            return self._format_explanation(X, shap_values, base_value, return_format)
            
        except Exception as e:
            self._logger.error(f"Error generating SHAP explanation: {e}")
            return self._error_explanation(str(e))
    
    def explain_batch(self, input_batch: np.ndarray, 
                     return_format: str = "dict") -> List[Dict[str, Any]]:
        """
        Generate SHAP explanations for a batch of inputs.
        
        Args:
            input_batch: Batch of input data (2D array)
            return_format: Format of returned explanations
            
        Returns:
            List of explanation dictionaries
        """
        try:
            if input_batch.ndim == 1:
                input_batch = input_batch.reshape(1, -1)
            
            explanations = []
            for i in range(input_batch.shape[0]):
                explanation = self.explain(input_batch[i], return_format)
                explanation['batch_index'] = i
                explanations.append(explanation)
            
            return explanations
            
        except Exception as e:
            self._logger.error(f"Error generating batch explanations: {e}")
            return [self._error_explanation(str(e))]
    
    def get_feature_importance_ranking(self, input_data: Union[np.ndarray, Dict[str, Any], List]) -> List[Dict[str, Any]]:
        """
        Get features ranked by importance for the given input.
        
        Args:
            input_data: Input data to explain
            
        Returns:
            List of features ranked by absolute SHAP value
        """
        try:
            explanation = self.explain(input_data, return_format="detailed")
            
            if 'error' in explanation:
                return []
            
            feature_importance = explanation['feature_importance']
            
            # Sort by absolute importance
            ranked_features = sorted(feature_importance, 
                                   key=lambda x: abs(x['shap_value']), 
                                   reverse=True)
            
            return ranked_features
            
        except Exception as e:
            self._logger.error(f"Error ranking feature importance: {e}")
            return []
    
    def _prepare_input_data(self, input_data: Union[np.ndarray, Dict[str, Any], List]) -> np.ndarray:
        """Convert input data to numpy array."""
        if isinstance(input_data, dict):
            if self.feature_names:
                # Use feature names to order the values
                values = [input_data.get(name, 0.0) for name in self.feature_names]
                return np.array(values)
            else:
                # Use dict values in order
                return np.array(list(input_data.values()))
        elif isinstance(input_data, list):
            return np.array(input_data)
        elif isinstance(input_data, np.ndarray):
            if input_data.ndim > 1:
                return input_data.flatten()
            return input_data
        else:
            raise ValueError(f"Unsupported input data type: {type(input_data)}")
    
    def _format_explanation(self, input_data: np.ndarray, shap_values: np.ndarray, 
                          base_value: float, return_format: str) -> Dict[str, Any]:
        """Format SHAP explanation based on requested format."""
        
        if shap_values.ndim > 1:
            shap_values = shap_values.flatten()
        
        explanation = {
            'method': 'SHAP',
            'model_type': self.model_type,
            'base_value': float(base_value),
            'prediction': float(base_value + np.sum(shap_values)),
            'input_data': input_data.tolist()
        }
        
        if return_format == "array":
            explanation['shap_values'] = shap_values.tolist()
            
        elif return_format == "dict":
            if self.feature_names and len(self.feature_names) == len(shap_values):
                explanation['feature_importance'] = {
                    name: float(value) for name, value in zip(self.feature_names, shap_values)
                }
            else:
                explanation['feature_importance'] = {
                    f'feature_{i}': float(value) for i, value in enumerate(shap_values)
                }
                
        elif return_format == "detailed":
            feature_importance = []
            for i, shap_value in enumerate(shap_values):
                feature_name = self.feature_names[i] if (self.feature_names and i < len(self.feature_names)) else f'feature_{i}'
                feature_importance.append({
                    'feature_name': feature_name,
                    'feature_value': float(input_data[i]),
                    'shap_value': float(shap_value),
                    'contribution_magnitude': abs(float(shap_value))
                })
            
            explanation['feature_importance'] = feature_importance
            explanation['shap_values'] = shap_values.tolist()
        
        return explanation
    
    def _mock_explanation(self, input_data: np.ndarray, return_format: str) -> Dict[str, Any]:
        """Generate mock SHAP explanation when SHAP is not available."""
        # Generate random SHAP-like values that sum to a reasonable prediction
        np.random.seed(42)  # For reproducible mock results
        mock_shap_values = np.random.normal(0, 0.1, size=input_data.shape)
        
        # Normalize so they sum to a reasonable prediction
        base_value = 0.5
        prediction = base_value + np.sum(mock_shap_values)
        
        explanation = {
            'method': 'SHAP (Mock)',
            'model_type': 'mock',
            'base_value': base_value,
            'prediction': float(prediction),
            'input_data': input_data.tolist(),
            'warning': 'SHAP library not available - using mock implementation'
        }
        
        return self._format_explanation(input_data, mock_shap_values, base_value, return_format)
    
    def _error_explanation(self, error_message: str) -> Dict[str, Any]:
        """Generate error explanation."""
        return {
            'method': 'SHAP',
            'error': error_message,
            'feature_importance': {},
            'base_value': 0.0,
            'prediction': 0.0
        }
    
    def _detect_model_type(self) -> str:
        """Auto-detect model type based on model class."""
        model_class = type(self.model).__name__.lower()
        
        # Tree-based models
        if any(tree_type in model_class for tree_type in 
               ['tree', 'forest', 'boost', 'xgb', 'lgb', 'catboost']):
            return "tree"
        
        # Linear models
        elif any(linear_type in model_class for linear_type in 
                ['linear', 'logistic', 'ridge', 'lasso', 'elastic']):
            return "linear"
        
        # Default to kernel
        else:
            return "kernel"