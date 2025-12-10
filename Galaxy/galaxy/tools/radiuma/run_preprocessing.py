from pysera.preprocessing import Preprocessor
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--method", choices=["resample", "shuffle"], required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

df = pd.read_csv(args.input)
prep = Preprocessor(df)

if args.method == "resample":
    result = prep.resample()
elif args.method == "shuffle":
    result = prep.shuffle()

result.to_csv(args.output, index=False)
