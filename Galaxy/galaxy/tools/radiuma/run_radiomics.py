# run_radiomics.py

import pysera
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True)
parser.add_argument("--mask", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--categories", default="glcm, glrlm")
parser.add_argument("--dimensions", default="1st, 2_5d, 3d")
parser.add_argument("--apply_preprocessing", default="True")
args = parser.parse_args()

# اطمینان از وجود مسیر خروجی
os.makedirs(args.output, exist_ok=True)

# اجرای PySERA
result = pysera.process_batch(
    image_input=args.image,
    mask_input=args.mask,
    output_path=args.output,
    categories=args.categories,
    dimensions=args.dimensions,
    apply_preprocessing=args.apply_preprocessing == "True"
)

# ذخیره خروجی ویژگی‌ها به CSV
if result["success"]:
    features = result["features_extracted"]
    features.to_csv(os.path.join(args.output, "radiomics_features.csv"), index=False)
else:
    print("Error:", result.get("error", "Unknown error"))
