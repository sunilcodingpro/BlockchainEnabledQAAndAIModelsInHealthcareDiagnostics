"""
Model Registry: Handles registration, lookup, and status of AI models.
"""

class ModelRegistry:
    def __init__(self):
        self.models = {}

    def register(self, model_id, metadata):
        self.models[model_id] = metadata
        return {"status": "registered", "model_id": model_id}

    def get(self, model_id):
        return self.models.get(model_id, None)

    def list_models(self):
        return list(self.models.keys())
