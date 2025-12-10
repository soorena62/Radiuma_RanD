import pandas as pd
from sklearn.utils import resample, shuffle

class Preprocessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def resample(self, replace=False):
        return resample(self.df, replace=replace)

    def shuffle(self):
        return shuffle(self.df)