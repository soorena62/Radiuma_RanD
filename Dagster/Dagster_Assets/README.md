# Dagster Assets Project

This project demonstrates how to orchestrate a radiomics workflow using **Dagster Assets**.
Assets are declarative building blocks that represent data or computation results. Each asset is materialized in sequence to produce reproducible outputs.

## Features
- Asset-based orchestration for radiomics extraction.
- Clear lineage: images → registration → fusion → filtering → mask alignment → feature extraction → final JSON/Excel.
- Custom IOManager to persist outputs as JSON files in the `artifacts/` directory.

## Requirements
- Python 3.10+
- Dagster
- SimpleITK
- Pandas
- PySERA

Install dependencies:
```bash
pip install dagster simpleitk pandas pysera
# Running:
python run_pipeline.py
