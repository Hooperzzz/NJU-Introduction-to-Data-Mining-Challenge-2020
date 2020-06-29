import os
import pickle
import random
import re

from comtypes.safearray import numpy
from gensim.models import word2vec

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from network_part.PredictByNetwork import PredictByNetwork


class DataGenerator:
    cur_loop_time = 0
    one_times = 1.0
    zero_times = 1.0

    def __init__(self, vectoried_test_path: str):
        with open(vectoried_test_path, "rb") as f:
            self.vectoried_test = pickle.load(f)
        self.dir_num = len(self.vectoried_test)
        self.dir_file_num = len(self.vectoried_test[0])

    def clone_test(self):
        i = random.randint(0, self.dir_num - 1)
        j1 = random.randint(0, self.dir_file_num - 1)
        j2 = j1
        while j1 == j2:
            j2 = random.randint(0, self.dir_file_num - 1)
        return {"input1": numpy.array([self.vectoried_test[i][j1]]),
                "input2": numpy.array([self.vectoried_test[i][j2]])}, numpy.array([[1]])

    def not_clone_test(self):
        i1 = random.randint(0, self.dir_num - 1)
        i2 = i1
        while i1 == i2:
            i2 = random.randint(0, self.dir_num - 1)
        j1 = random.randint(0, self.dir_file_num - 1)
        j2 = random.randint(0, self.dir_file_num - 1)
        return {"input1": numpy.array([self.vectoried_test[i1][j1]]),
                "input2": numpy.array([self.vectoried_test[i2][j2]])}, numpy.array([[0]])


def use_exe_save_and_analysis_str_and_network(
        vectoried_test_path: str,
        model_path: str, test_dir_path: str,
        test_len: int, word_len: int, thresh_num: float,
        times: int, is_test_clone: bool) -> int:
    """
    用于获得写报告的数据
    """
    predict_by_network = PredictByNetwork(model_path, test_dir_path,
                                          test_len, word_len, thresh_num)
    dg = DataGenerator(vectoried_test_path)
    clone_num = 0
    for _ in range(times):
        if is_test_clone:
            i = dg.clone_test()
        else:
            i = dg.not_clone_test()
        if predict_by_network.predict1(i[0]["input1"], i[0]["input2"]):
            clone_num += 1
    return clone_num


if __name__ == "__main__":
    use_exe_save_and_analysis_str_and_network("../build/treated_str_train",
                                              "../build/model", "../build/treated_str_test",
                                              300, 20, 0.60,
                                              1000, False)
