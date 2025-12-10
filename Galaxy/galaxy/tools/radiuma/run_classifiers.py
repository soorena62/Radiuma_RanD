from pysera.classifier import Classifier
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--x_train", required=True)
parser.add_argument("--y_train", required=True)
parser.add_argument("--x_val", required=True)
parser.add_argument("--y_val", required=True)
parser.add_argument("--algorithm", choices=["LogisticRegression", "KNeighborsClassifier"], required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

X_train = pd.read_csv(args.x_train)
y_train = pd.read_csv(args.y_train)
X_val = pd.read_csv(args.x_val)
y_val = pd.read_csv(args.y_val)

clf = Classifier(X_train, y_train, X_val, y_val)
result = clf.run(algorithm=args.algorithm)
pd.DataFrame(result).to_csv(args.output, index=False)
