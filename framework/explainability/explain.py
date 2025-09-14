"""
Explainability: Uses SHAP/LIME for model explanation.
"""

import shap
import lime.lime_tabular

class Explainability:
    def __init__(self, model):
        self.model = model

    def shap_explain(self, X):
        explainer = shap.Explainer(self.model)
        return explainer(X)

    def lime_explain(self, X, training_data):
        explainer = lime.lime_tabular.LimeTabularExplainer(training_data)
        return explainer.explain_instance(X, self.model.predict_proba)
