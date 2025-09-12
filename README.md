# Blockchain-Enabled Quality Assurance for AI Models in Healthcare Diagnostics

This project implements a simulation and prototype of a blockchain-enabled quality assurance framework for AI models in healthcare diagnostics. The system leverages Hyperledger Fabric to create immutable audit trails for model versions, data provenance, diagnostic decisions, and explainability reports.

## Features

- **Model Registry:** Tracks cryptographic hashes of model files and version differentials.
- **Data Provenance Tracking:** Ensures all training and validation data is auditable and tamper-proof.
- **Decision Audit Trail:** Logs all clinical AI inferences with explainability data.
- **Explainability Integration:** Supports SHAP/LIME explanations and domain heuristics.
- **Regulatory Interface:** Generates FDA/EU MDR-compliant documentation from blockchain records.
- **Simulation Harness:** Models performance drift, deployment, audits, and regulatory queries.

## Architecture

See [docs/architecture.md](docs/architecture.md).

## Getting Started

1. Install requirements:
    ```bash
    pip install -r requirements.txt
    ```
2. Configure your Hyperledger Fabric environment in `framework/config.py`.
3. Run simulation:
    ```bash
    python -m framework.simulation.simulator
    ```

## License

See [LICENSE](LICENSE).