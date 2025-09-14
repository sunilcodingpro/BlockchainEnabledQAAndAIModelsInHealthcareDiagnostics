from framework.blockchain.chaincode import BlockchainQAChaincode

def test_register_model():
    chaincode = BlockchainQAChaincode()
    model_id = "model-xyz"
    metadata = {"type": "QA", "version": "2.0"}
    result = chaincode.register_model(model_id, metadata)
    assert result["status"] == "success"
    assert chaincode.models[model_id] == metadata
