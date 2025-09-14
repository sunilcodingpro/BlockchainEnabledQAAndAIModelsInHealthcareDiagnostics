from framework.model_registry.registry import ModelRegistry

def test_register_and_get():
    registry = ModelRegistry()
    model_id = "model-abc"
    metadata = {"type": "diagnostic", "version": "1.0"}
    result = registry.register(model_id, metadata)
    assert result["status"] == "registered"
    assert registry.get(model_id) == metadata
