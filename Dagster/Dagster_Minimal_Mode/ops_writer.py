from dagster import op

@op
def write_report(features):
    with open("artifacts/final_report.txt", "w", encoding="utf-8") as f:
        f.write("# Radiomics Report\n")
        f.write(f"Extracted {len(features)} features\n")
        for k, v in features.items():
            f.write(f"{k},{v}\n")
