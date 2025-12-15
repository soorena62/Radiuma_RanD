## README For **Dagster_Minimal_Mode**

```markdown
# Dagster Minimal Mode Project

This project demonstrates the radiomics workflow in **Dagster Minimal Mode**.
Minimal Mode is a lightweight configuration of Dagster, focusing on simplicity and reduced overhead.

## Features
- Minimal orchestration setup for radiomics extraction.
- Direct execution of ops/assets without full Dagster deployment.
- Simplified configuration for quick testing and prototyping.

## Requirements
- Python 3.10+
- Dagster (minimal mode enabled)
- SimpleITK
- Pandas
- PySERA

Install dependencies:
```bash
pip install dagster simpleitk pandas pysera

# Running:
python radiuma_pipeline.py
