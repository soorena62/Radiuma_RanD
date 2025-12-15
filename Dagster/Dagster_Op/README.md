## README **Dagster_op**

```markdown
# Dagster Ops Project

This project demonstrates the same radiomics workflow using **Dagster Ops**.
Ops are imperative functions that define computation steps. They are connected in a job graph to form the pipeline.

## Features
- Op-based orchestration for radiomics extraction.
- Explicit control of execution order via job definitions.
- Each op corresponds to one stage: image reading, registration, fusion, filtering, mask registration, feature extraction, and writing results.

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
dagster job execute -f dagster_op_runner.py
Or:
run.bat
