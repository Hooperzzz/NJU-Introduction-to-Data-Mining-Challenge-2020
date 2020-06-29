import os
import pickle

from tensorflow import keras
from comtypes.safearray import numpy
from gensim.models import word2vec
from network_part.UseNetwork import discrete_code, make_test_vectoried


class PredictByNetwork:
    def __init__(self, model_path: str, test_vectoried_test_path: str,
                 test_len: int, word_len: int, thresh_num: float):
        self.model = keras.models.load_model(model_path)
        with open(test_vectoried_test_path, "rb") as f:
            self.test_vectoried_test = pickle.load(f)["../test/test"]
        self.test_len = test_len
        self.word_len = word_len
        self.thresh_num = thresh_num

    def predict(self, name1: str, name2: str) -> bool:
        mtv1 = self.test_vectoried_test[name1]
        mtv2 = self.test_vectoried_test[name2]
        predicts = self.model.predict([mtv1, mtv2], batch_size=1)
        res = True if predicts[0][0] > self.thresh_num else False
        return res

    def predict1(self, input1, input2):
        predicts = self.model.predict([input1, input2], batch_size=1)
        res = True if predicts[0][0] > self.thresh_num else False
        return res
