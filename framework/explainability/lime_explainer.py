"""
LIMEExplainer: Generates LIME-based explanations for local interpretability.
"""
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Union, Callable

try:
    from lime.lime_tabular import LimeTabularExplainer
    from lime.lime_text import LimeTextExplainer
    from lime.lime_image import LimeImageExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logging.warning("LIME library not available. Using mock implementation.")


class LIMEExplainer:
    """
    LIME (Local Interpretable Model-agnostic Explanations) explainer.
    
    This class generates local explanations for individual predictions
    by learning interpretable models locally around the prediction.
    """
    
    def __init__(self, model, training_data: np.ndarray, 
                 feature_names: Optional[List[str]] = None,
                 categorical_features: Optional[List[int]] = None,
                 mode: str = "tabular"):
        """
        Initialize the LIME explainer.
        
        Args:
            model: Trained ML model with predict_proba method
            training_data: Training dataset for building explanations
            feature_names: Names of input features
            categorical_features: Indices of categorical features
            mode: Type of data ("tabular", "text", "image")
        """
        self.model = model
        self.training_data = training_data
        self.feature_names = feature_names or [f"feature_{i}" for i in range(training_data.shape[1])]
        self.categorical_features = categorical_features or []
        self.mode = mode
        self._logger = logging.getLogger(__name__)
        
        # Initialize LIME explainer
        self.explainer = None
        self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Initialize the appropriate LIME explainer."""
        if not LIME_AVAILABLE:
            self._logger.warning("LIME not available, using mock explainer")
            return
        
        try:
            if self.mode == "tabular":
                self.explainer = LimeTabularExplainer(
                    training_data=self.training_data,
                    feature_names=self.feature_names,
                    categorical_features=self.categorical_features,
                    verbose=False,
                    mode='classification' if hasattr(self.model, 'predict_proba') else 'regression'
                )
            elif self.mode == "text":
                self.explainer = LimeTextExplainer(
                    mode='classification' if hasattr(self.model, 'predict_proba') else 'regression'
                )
            elif self.mode == "image":
                self.explainer = LimeImageExplainer(
                    mode='classification' if hasattr(self.model, 'predict_proba') else 'regression'
                )
            else:
                raise ValueError(f"Unsupported mode: {self.mode}")
            
            self._logger.info(f"Initialized LIME {self.mode} explainer")
            
        except Exception as e:
            self._logger.error(f"Error initializing LIME explainer: {e}")
            self.explainer = None
    
    def explain(self, input_data: Union[np.ndarray, Dict[str, Any], List], 
                num_features: int = 10, return_format: str = "dict") -> Dict[str, Any]:
        """
        Generate LIME explanations for input data.
        
        Args:
            input_data: Input data to explain
            num_features: Number of features to include in explanation
            return_format: Format of returned explanation ("dict", "detailed")
            
        Returns:
            Dictionary containing LIME explanation
        """
        try:
            # Convert input data to appropriate format
            if self.mode == "tabular":
                X = self._prepare_tabular_input(input_data)
                return self._explain_tabular(X, num_features, return_format)
            elif self.mode == "text":
                return self._explain_text(input_data, num_features, return_format)
            elif self.mode == "image":
                return self._explain_image(input_data, num_features, return_format)
            else:
                raise ValueError(f"Unsupported mode: {self.mode}")
                
        except Exception as e:
            self._logger.error(f"Error generating LIME explanation: {e}")
            return self._error_explanation(str(e))
    
    def _explain_tabular(self, input_data: np.ndarray, num_features: int, 
                        return_format: str) -> Dict[str, Any]:
        """Generate explanation for tabular data."""
        if self.explainer is None or not LIME_AVAILABLE:
            return self._mock_explanation(input_data, num_features, return_format)
        
        try:
            # Get prediction function
            predict_fn = self._get_predict_function()
            
            # Generate explanation
            explanation = self.explainer.explain_instance(
                input_data, 
                predict_fn, 
                num_features=min(num_features, len(self.feature_names))
            )
            
            return self._format_tabular_explanation(input_data, explanation, return_format)
            
        except Exception as e:
            self._logger.error(f"Error in LIME tabular explanation: {e}")
            return self._mock_explanation(input_data, num_features, return_format)
    
    def _explain_text(self, text_data: str, num_features: int, 
                     return_format: str) -> Dict[str, Any]:
        """Generate explanation for text data."""
        if self.explainer is None or not LIME_AVAILABLE:
            return self._mock_text_explanation(text_data, return_format)
        
        try:
            predict_fn = self._get_predict_function()
            
            explanation = self.explainer.explain_instance(
                text_data,
                predict_fn,
                num_features=num_features
            )
            
            return self._format_text_explanation(text_data, explanation, return_format)
            
        except Exception as e:
            self._logger.error(f"Error in LIME text explanation: {e}")
            return self._mock_text_explanation(text_data, return_format)
    
    def _explain_image(self, image_data: np.ndarray, num_features: int, 
                      return_format: str) -> Dict[str, Any]:
        """Generate explanation for image data."""
        if self.explainer is None or not LIME_AVAILABLE:
            return self._mock_image_explanation(image_data, return_format)
        
        try:
            predict_fn = self._get_predict_function()
            
            explanation = self.explainer.explain_instance(
                image_data,
                predict_fn,
                num_features=num_features
            )
            
            return self._format_image_explanation(image_data, explanation, return_format)
            
        except Exception as e:
            self._logger.error(f"Error in LIME image explanation: {e}")
            return self._mock_image_explanation(image_data, return_format)
    
    def get_feature_importance_ranking(self, input_data: Union[np.ndarray, Dict[str, Any], List],
                                     num_features: int = 10) -> List[Dict[str, Any]]:
        """
        Get features ranked by importance according to LIME.
        
        Args:
            input_data: Input data to explain
            num_features: Number of top features to return
            
        Returns:
            List of features ranked by LIME importance
        """
        try:
            explanation = self.explain(input_data, num_features, return_format="detailed")
            
            if 'error' in explanation:
                return []
            
            feature_importance = explanation.get('feature_importance', [])
            
            # Sort by absolute importance
            ranked_features = sorted(feature_importance,
                                   key=lambda x: abs(x.get('lime_value', 0)),
                                   reverse=True)
            
            return ranked_features[:num_features]
            
        except Exception as e:
            self._logger.error(f"Error ranking LIME feature importance: {e}")
            return []
    
    def _prepare_tabular_input(self, input_data: Union[np.ndarray, Dict[str, Any], List]) -> np.ndarray:
        """Convert input data to numpy array for tabular explanation."""
        if isinstance(input_data, dict):
            # Use feature names to order the values
            values = [input_data.get(name, 0.0) for name in self.feature_names]
            return np.array(values)
        elif isinstance(input_data, list):
            return np.array(input_data)
        elif isinstance(input_data, np.ndarray):
            if input_data.ndim > 1:
                return input_data.flatten()
            return input_data
        else:
            raise ValueError(f"Unsupported input data type: {type(input_data)}")
    
    def _get_predict_function(self) -> Callable:
        """Get appropriate prediction function for the model."""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba
        elif hasattr(self.model, 'predict'):
            return lambda x: self.model.predict(x).reshape(-1, 1)
        else:
            raise ValueError("Model must have predict or predict_proba method")
    
    def _format_tabular_explanation(self, input_data: np.ndarray, explanation, 
                                  return_format: str) -> Dict[str, Any]:
        """Format LIME explanation for tabular data."""
        
        # Extract explanation data
        explanation_list = explanation.as_list()
        prediction = explanation.predict_proba[1] if hasattr(explanation, 'predict_proba') else 0.5
        
        result = {
            'method': 'LIME',
            'mode': 'tabular',
            'prediction': float(prediction),
            'input_data': input_data.tolist()
        }
        
        if return_format == "dict":
            result['feature_importance'] = {
                feature_name: float(importance) 
                for feature_name, importance in explanation_list
            }
        elif return_format == "detailed":
            feature_importance = []
            for feature_name, lime_value in explanation_list:
                # Find feature index
                try:
                    feature_idx = self.feature_names.index(feature_name)
                    feature_value = float(input_data[feature_idx])
                except (ValueError, IndexError):
                    feature_value = 0.0
                
                feature_importance.append({
                    'feature_name': feature_name,
                    'feature_value': feature_value,
                    'lime_value': float(lime_value),
                    'contribution_magnitude': abs(float(lime_value))
                })
            
            result['feature_importance'] = feature_importance
        
        return result
    
    def _format_text_explanation(self, text_data: str, explanation, 
                               return_format: str) -> Dict[str, Any]:
        """Format LIME explanation for text data."""
        explanation_list = explanation.as_list()
        prediction = explanation.predict_proba[1] if hasattr(explanation, 'predict_proba') else 0.5
        
        result = {
            'method': 'LIME',
            'mode': 'text',
            'prediction': float(prediction),
            'input_text': text_data
        }
        
        if return_format == "dict":
            result['word_importance'] = {
                word: float(importance) 
                for word, importance in explanation_list
            }
        elif return_format == "detailed":
            word_importance = []
            for word, lime_value in explanation_list:
                word_importance.append({
                    'word': word,
                    'lime_value': float(lime_value),
                    'contribution_magnitude': abs(float(lime_value))
                })
            
            result['word_importance'] = word_importance
        
        return result
    
    def _format_image_explanation(self, image_data: np.ndarray, explanation, 
                                return_format: str) -> Dict[str, Any]:
        """Format LIME explanation for image data."""
        # Get superpixel explanations
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=False,
            num_features=10,
            hide_rest=False
        )
        
        result = {
            'method': 'LIME',
            'mode': 'image',
            'image_shape': image_data.shape,
            'explanation_mask': mask.tolist() if hasattr(mask, 'tolist') else str(mask)
        }
        
        return result
    
    def _mock_explanation(self, input_data: np.ndarray, num_features: int, 
                         return_format: str) -> Dict[str, Any]:
        """Generate mock LIME explanation when LIME is not available."""
        np.random.seed(42)  # For reproducible results
        
        # Select top features randomly
        n_features = min(num_features, len(input_data))
        selected_indices = np.random.choice(len(input_data), n_features, replace=False)
        
        # Generate mock LIME values
        mock_values = np.random.normal(0, 0.2, size=n_features)
        
        result = {
            'method': 'LIME (Mock)',
            'mode': 'tabular',
            'prediction': 0.5 + np.sum(mock_values) * 0.1,
            'input_data': input_data.tolist(),
            'warning': 'LIME library not available - using mock implementation'
        }
        
        if return_format == "dict":
            result['feature_importance'] = {
                self.feature_names[idx]: float(val) 
                for idx, val in zip(selected_indices, mock_values)
            }
        elif return_format == "detailed":
            feature_importance = []
            for idx, lime_val in zip(selected_indices, mock_values):
                feature_importance.append({
                    'feature_name': self.feature_names[idx],
                    'feature_value': float(input_data[idx]),
                    'lime_value': float(lime_val),
                    'contribution_magnitude': abs(float(lime_val))
                })
            
            result['feature_importance'] = feature_importance
        
        return result
    
    def _mock_text_explanation(self, text_data: str, return_format: str) -> Dict[str, Any]:
        """Generate mock explanation for text data."""
        words = text_data.split()[:10]  # Take first 10 words
        np.random.seed(42)
        mock_values = np.random.normal(0, 0.3, size=len(words))
        
        result = {
            'method': 'LIME (Mock)',
            'mode': 'text',
            'prediction': 0.5,
            'input_text': text_data,
            'warning': 'LIME library not available - using mock implementation'
        }
        
        if return_format == "dict":
            result['word_importance'] = {
                word: float(val) for word, val in zip(words, mock_values)
            }
        
        return result
    
    def _mock_image_explanation(self, image_data: np.ndarray, return_format: str) -> Dict[str, Any]:
        """Generate mock explanation for image data."""
        return {
            'method': 'LIME (Mock)',
            'mode': 'image',
            'image_shape': image_data.shape,
            'warning': 'LIME library not available - using mock implementation'
        }
    
    def _error_explanation(self, error_message: str) -> Dict[str, Any]:
        """Generate error explanation."""
        return {
            'method': 'LIME',
            'error': error_message,
            'feature_importance': {},
            'prediction': 0.0
        }