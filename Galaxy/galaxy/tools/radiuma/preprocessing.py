from sklearn.utils import resample, shuffle

class preprocessing:

    def __init__(self,X):
        self.X = X



    # def SelectAlg(self):
    #     import json
    #     f = open('data.json')
    #     JSONdata = json.load(f)
    #     f.close()
    #     algn = self.AlgName
    #     if self.AlgName == "resample":

    #         self.Vs_resample(
    #             JSONdata[algn][0],
    #             JSONdata[algn][1]
    #         )
    #     elif self.AlgName == "shuffle":
    #         self.Vs_shuffle(
    #             JSONdata[algn][0],
    #             JSONdata[algn][1]
    #         )
    #     else:
    #         print("Algorithm name is wrong")



    def Vs_resampling(self,str=False,rep=False):

        if str:
            x2 = resample(self.X , replace=rep)
        else:
            x2 = resample(self.X , replace=rep)

        return x2



    def Vs_shuffling(self):

        x2 = shuffle(self.X)

        return x2
