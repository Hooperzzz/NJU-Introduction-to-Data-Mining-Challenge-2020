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


# 参考代码：
#   https://blog.csdn.net/qq_35273499/article/details/79098689
#   https://www.jianshu.com/p/a649b568e8fa
#   https://radimrehurek.com/gensim/models/word2vec.html
# 遇到的报错：
#   TypeError: objectof type 'xxx' has no len(): https://blog.csdn.net/weixin_40818267/article/details/104818917
#   EOFError: Ran out of input（尚未处理）: https://stackoverflow.com/questions/44704086/eoferror-ran-out-of-input-inside-a-class
#   cudaGetDevice() failed. Status: CUDA driver version is insufficient for CUDA runtime version: 安装GEFORCE EXPERIENCE


def discrete_code(code: str) -> list:
    # 对代码进行分词
    all_word = []
    word = re.split(
        "(!|%|\\^|&|\\*|\\(|\\)|-|\\+|=|\\{|\\}|[|]|:|;|<|,|>|\\.|\\?|/|\n|\t|\r|'|\"| )",
        code)
    while "\n" in word:
        word.remove("\n")
    while "\t" in word:
        word.remove("\t")
    while "\r" in word:
        word.remove("\r")
    while " " in word:
        word.remove(" ")
    while "" in word:
        word.remove("")
    i, end = 0, len(word)
    while i < end:
        if word[i].isdigit():
            all_word.append("@NUM")
        elif word[0] in "`~!^&*()-+={}[]|\\:;\"'<,>.?/":
            pass
        elif word[:3] == "VAR":
            all_word.append(word[i])
            # all_word.append("@VAR")
        elif word[i] == '"':
            all_word.append("@STR")
            i += 1
            while word[i] != '"':
                if word[i] == "\\":
                    i += 1
                i += 1
        elif word[i] == "'":
            all_word.append("@CHAR")
            i += 1
            while word[i] != "'":
                if word[i] == "\\":
                    i += 1
                i += 1
        else:
            all_word.append(word[i])
        i += 1
    return all_word


def get_divided_code(train_path: str, save_path: str) -> None:
    # 在代码中不同成分之间添加空格
    all_word = []
    for root, dirs, files in os.walk(train_path):
        for name in files:
            with open(os.path.join(root, name), "r") as f:
                code = f.read()
            all_word.append(discrete_code(code))
            # break
        # if all_word:
        #     break
    # tokenizer = text.Tokenizer(filters="", lower=False, char_level=False)
    # tokenizer.fit_on_texts(all_word)
    # # i = tokenizer.word_counts
    # with open(save_path, "wb") as f:
    #     pickle.dump(tokenizer, f)
    with open(save_path, "wb") as f:
        pickle.dump(all_word, f)

# 一下为使用word2vec方法的代码


def get_divided_code_with_min_count(divided_code_path: str, save_tokenizer_path: str, save_file_path: str,
                                    min_count: int) -> None:
    # 首先所有的出现次数小于min_count的词替换成同一个词
    with open(divided_code_path, "rb") as f:
        all_word = pickle.load(f)
    tokenizer = text.Tokenizer(filters="", lower=False, char_level=False)
    tokenizer.fit_on_texts(all_word)
    # i = tokenizer.word_counts
    with open(save_tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)
    with open(save_file_path, "w") as f:
        for s in all_word:
            for w in s:
                if tokenizer.word_counts[w] >= min_count:
                    f.write(w + " ")
                else:
                    f.write("@OTHER ")
            f.write("\n")
    return


def get_word_vec(divided_code_path: str, save_path: str, word_len: int) -> None:
    # 根据tokenizer来表示每一个测试
    sentence = word2vec.Text8Corpus(divided_code_path)

    model = word2vec.Word2Vec(sentences=sentence, size=word_len, hs=1, window=3)
    model.save(save_path)


def get_divided_code_dict(train_path: str, save_path: str) -> None:
    # 算是一个用来补救之前失误的函数，按照字典的方式存储code
    # (傻了，应该是不需要字典，只要是数组就可以了的，故此函数废弃)
    all_word_dict = dict()
    for root, dirs, files in os.walk(train_path):
        for name in files:
            with open(os.path.join(root, name), "r") as f:
                code = f.read()
            if root not in all_word_dict.keys():
                all_word_dict[root] = dict()
            all_word_dict[root][name] = discrete_code(code)
    with open(save_path, "wb") as f:
        pickle.dump(all_word_dict, f)


def make_test_vectoried(model, s: list, test_len: int, word_len: int) -> list:
    sentence = []
    for w in s:
        try:
            v = model.wv.word_vec(w)
        except KeyError:
            v = model.wv.word_vec("@OTHER")
        sentence.append(v)
    add_sentence = []
    if len(sentence) < test_len:
        add_sentence = [[0 for i in range(word_len)] for j in range(test_len - len(sentence))]
        add_sentence.extend(sentence)
    elif len(sentence) == test_len:
        add_sentence.extend(sentence)
    else:
        add_sentence = sentence[len(sentence) - test_len:]
    return add_sentence


def get_vectoried_test(divided_code_dict_path: str, word2vec_model_path: str, save_path: str,
                       test_len: int, word_len: int) -> None:
    # 应该是需要数组，而不是字典，后面把生成字典的部分改为生成数组
    model = word2vec.Word2Vec.load(word2vec_model_path)
    with open(divided_code_dict_path, "rb") as f:
        all_word_dict = pickle.load(f)
    # vectoried_test = dict()
    vectoried_test = []
    for dirs, sub_list in all_word_dict.items():
        one_dir_sentence = []
        for name, s in sub_list.items():
            # sentence = []
            # for w in s:
            #     try:
            #         v = model.wv.word_vec(w)
            #     except KeyError:
            #         v = model.wv.word_vec("@OTHER")
            #     sentence.append(v)
            # add_sentence = []
            # if len(sentence) < test_len:
            #     add_sentence = [[0 for i in range(word_len)] for j in range(test_len - len(sentence))]
            #     add_sentence.extend(sentence)
            # elif len(sentence) == test_len:
            #     add_sentence.extend(sentence)
            # else:
            #     add_sentence = sentence[len(sentence) - test_len:]
            add_sentence = make_test_vectoried(model, s, test_len, word_len)
            # if dirs not in vectoried_test.keys():
            #     vectoried_test[dirs] = dict()
            # vectoried_test[dirs][name] = add_sentence
            one_dir_sentence.append(add_sentence)
        if one_dir_sentence:
            vectoried_test.append(one_dir_sentence)
    with open(save_path, "wb") as f:
        pickle.dump(vectoried_test, f)
    return


def get_test_vectoried_test(test_dir_path: str, word2vec_model_path: str, save_path: str,
                            test_len: int, word_len: int):
    # 把测试集向量化并存入字典中，保存下来
    model = word2vec.Word2Vec.load(word2vec_model_path)
    all_word_dict = dict()
    for root, dirs, files in os.walk(test_dir_path):
        for name in files:
            with open(os.path.join(root, name), "r") as f:
                code = f.read()
            dc = discrete_code(code)
            add_sentence = make_test_vectoried(model, dc, test_len, word_len)
            if root not in all_word_dict.keys():
                all_word_dict[root] = dict()
            all_word_dict[root][name.split(".")[0]] = numpy.array([add_sentence])
    with open(save_path, "wb") as f:
        pickle.dump(all_word_dict, f)


def get_trained_model(vectoried_test_path: str, save_path: str,
                      test_len: int, word_len: int,
                      lstm_output_len: int, lstm_dropout: float, lstm_recurrent_dropout: float,
                      dropout_rate: float,
                      dense_unit: int,
                      construct_pair_num: int, one_zero_rate: float, epoch_num: int
                      ) -> None:
    def get_model():
        lstm_layer = layers.LSTM(units=lstm_output_len, dropout=lstm_dropout, recurrent_dropout=lstm_recurrent_dropout)

        input1 = layers.Input(shape=(test_len, word_len), dtype=tf.float32, name="input1")
        y1 = lstm_layer(input1)
        input2 = layers.Input(shape=(test_len, word_len), dtype=tf.float32, name="input2")
        y2 = lstm_layer(input2)

        model_layer = layers.concatenate([y1, y2], name="concatenate1")
        model_layer = layers.Dropout(dropout_rate, name="Dropout1")(model_layer)
        model_layer = layers.BatchNormalization(name="BatchNormalization1")(model_layer)
        model_layer = layers.Dense(units=dense_unit, activation='relu', name="Dense1")(model_layer)
        model_layer = layers.Dropout(dropout_rate, name="Dropout2")(model_layer)
        model_layer = layers.BatchNormalization(name="BatchNormalization2")(model_layer)
        model_layer = layers.Dense(units=1, activation='sigmoid', name="Dense2")(model_layer)

        func_model = Model(inputs=[input1, input2], outputs=model_layer)
        func_model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])
        func_model.summary()
        return func_model

    class DataGenerator(keras.utils.Sequence):
        cur_loop_time = 0
        one_times = 1.0
        zero_times = 1.0

        def __init__(self):
            with open(vectoried_test_path, "rb") as f:
                self.vectoried_test = pickle.load(f)
            self.construct_pair_num = construct_pair_num
            self.one_zero_rate = one_zero_rate
            self.dir_num = len(self.vectoried_test)
            self.dir_file_num = len(self.vectoried_test[0])

        def __len__(self):
            return self.construct_pair_num

        def __getitem__(self, item):
            if self.one_times / self.zero_times <= self.one_zero_rate:
                self.one_times += 1
                i = random.randint(0, self.dir_num-1)
                j1 = random.randint(0, self.dir_file_num-1)
                j2 = j1
                while j1 == j2:
                    j2 = random.randint(0, self.dir_file_num - 1)
                return {"input1": numpy.array([self.vectoried_test[i][j1]]),
                        "input2": numpy.array([self.vectoried_test[i][j2]])}, numpy.array([[1]])
            else:
                self.zero_times += 1
                i1 = random.randint(0, self.dir_num - 1)
                i2 = i1
                while i1 == i2:
                    i2 = random.randint(0, self.dir_num - 1)
                j1 = random.randint(0, self.dir_file_num-1)
                j2 = random.randint(0, self.dir_file_num-1)
                return {"input1": numpy.array([self.vectoried_test[i1][j1]]),
                        "input2": numpy.array([self.vectoried_test[i2][j2]])}, numpy.array([[0]])

    model = get_model()
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    # 这个函数的文档：https://keras.io/zh/models/model/
    hist = model.fit_generator(generator=DataGenerator(), epochs=epoch_num, verbose=1, use_multiprocessing=False)
    model.save(save_path)
    return

# 一下为通过embedding层来映射的方法


def get_tokenizer_and_train(divided_code_path: str, save_tokenizer_path: str, save_file_path: str,
                            min_count: int, max_len: int) -> None:
    with open(divided_code_path, "rb") as f:
        all_word = pickle.load(f)
    tokenizer = text.Tokenizer(filters="", lower=False, char_level=False)
    tokenizer.fit_on_texts(all_word)
    treated_str_train = []

    count = 0
    dir_file_list = []
    final_train = []
    for s in all_word:
        sentence = []
        for w in s:
            if tokenizer.word_counts[w] >= min_count:
                sentence.append(w)
            else:
                sentence.append("@OTHER")
        treated_str_train.append(sentence)
        dir_file_list.append(sentence)
        count += 1
        if count == 500:
            final_train.append(dir_file_list)
            dir_file_list = []
            count = 0
    save_tokenizer = text.Tokenizer(filters="", lower=False, char_level=False)
    save_tokenizer.fit_on_texts(treated_str_train)

    for i in range(len(final_train)):
        term1 = save_tokenizer.texts_to_sequences(final_train[i])
        final_train[i] = pad_sequences(term1, maxlen=max_len)
    with open(save_tokenizer_path, "wb") as f:
        pickle.dump(save_tokenizer, f)
    with open(save_file_path, "wb") as f:
        pickle.dump(final_train, f)
    return


def get_tokenized_test(tokenizer_path: str, test_dir_path: str, save_path: str, max_len: int) -> None:
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    all_word_dict = dict()
    for root, dirs, files in os.walk(test_dir_path):
        for name in files:
            with open(os.path.join(root, name), "r") as f:
                code = f.read()
            dc = discrete_code(code)
            sentence = []
            for w in dc:
                if w in tokenizer.word_counts.keys():
                    sentence.append(w)
                else:
                    sentence.append("@OTHER")
            if root not in all_word_dict.keys():
                all_word_dict[root] = dict()
            all_word_dict[root][name.split(".")[0]] = pad_sequences(tokenizer.texts_to_sequences([sentence]),
                                                                    maxlen=max_len)
    with open(save_path, "wb") as f:
        pickle.dump(all_word_dict, f)


def get_trained_model_using_embed(vectoried_test_path: str, save_path: str, tokenizer_path: str,
                      test_len: int,
                      embedding_output_size: int,
                      lstm_output_len: int, lstm_dropout: float, lstm_recurrent_dropout: float,
                      dropout_rate: float,
                      dense_unit: int,
                      construct_pair_num: int, one_zero_rate: float, epoch_num: int
                      ) -> None:
    def get_model():
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
        embedding_layer = layers.Embedding(input_dim=len(tokenizer.word_index)+1,
                                           output_dim=embedding_output_size,
                                           input_length=test_len)
        lstm_layer = layers.LSTM(units=lstm_output_len, dropout=lstm_dropout, recurrent_dropout=lstm_recurrent_dropout)

        input1 = layers.Input(shape=(test_len), dtype=tf.float32, name="input1")
        e1 = embedding_layer(input1)
        y1 = lstm_layer(e1)
        input2 = layers.Input(shape=(test_len), dtype=tf.float32, name="input2")
        e2 = embedding_layer(input2)
        y2 = lstm_layer(e2)

        model_layer = layers.concatenate([y1, y2], name="concatenate1")
        model_layer = layers.Dropout(dropout_rate, name="Dropout1")(model_layer)
        model_layer = layers.BatchNormalization(name="BatchNormalization1")(model_layer)
        model_layer = layers.Dense(units=dense_unit, activation='relu', name="Dense1")(model_layer)
        model_layer = layers.Dropout(dropout_rate, name="Dropout2")(model_layer)
        model_layer = layers.BatchNormalization(name="BatchNormalization2")(model_layer)
        model_layer = layers.Dense(units=1, activation='sigmoid', name="Dense2")(model_layer)

        func_model = Model(inputs=[input1, input2], outputs=model_layer)
        func_model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])
        func_model.summary()
        return func_model

    class DataGenerator(keras.utils.Sequence):
        cur_loop_time = 0
        one_times = 1.0
        zero_times = 1.0

        def __init__(self):
            with open(vectoried_test_path, "rb") as f:
                self.vectoried_test = pickle.load(f)
            self.construct_pair_num = construct_pair_num
            self.one_zero_rate = one_zero_rate
            self.dir_num = len(self.vectoried_test)
            self.dir_file_num = len(self.vectoried_test[0])

        def __len__(self):
            return self.construct_pair_num

        def __getitem__(self, item):
            if self.one_times / self.zero_times <= self.one_zero_rate:
                self.one_times += 1
                i = random.randint(0, self.dir_num-1)
                j1 = random.randint(0, self.dir_file_num-1)
                j2 = j1
                while j1 == j2:
                    j2 = random.randint(0, self.dir_file_num - 1)
                return {"input1": numpy.array([self.vectoried_test[i][j1]]),
                        "input2": numpy.array([self.vectoried_test[i][j2]])}, numpy.array([[1]])
            else:
                self.zero_times += 1
                i1 = random.randint(0, self.dir_num - 1)
                i2 = i1
                while i1 == i2:
                    i2 = random.randint(0, self.dir_num - 1)
                j1 = random.randint(0, self.dir_file_num-1)
                j2 = random.randint(0, self.dir_file_num-1)
                return {"input1": numpy.array([self.vectoried_test[i1][j1]]),
                        "input2": numpy.array([self.vectoried_test[i2][j2]])}, numpy.array([[0]])

    model = get_model()
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    # 这个函数的文档：https://keras.io/zh/models/model/
    hist = model.fit_generator(generator=DataGenerator(), epochs=epoch_num, verbose=1, use_multiprocessing=False)
    model.save(save_path)
    return


if __name__ == "__main__":
    # get_divided_code("../train", "../build/divided_code")
    # 一下是使用word2vec方法的代码
    # get_divided_code_with_min_count("../build/divided_code", "../build/tokenizer", "../build/divided_code.txt", 5)
    # get_word_vec("../build/divided_code.txt", "../build/word2vec_model", 20)
    # get_divided_code_dict("../train", "../build/divided_code_dict")
    # get_vectoried_test("../build/divided_code_dict", "../build/word2vec_model", "../build/vectoried_test", 500, 20)
    # get_test_vectoried_test("../test/test", "../build/word2vec_model", "../build/test_vectoried_test", 500, 20)
    # get_trained_model("../build/vectoried_test", "../build/model",
    #                   500, 20,
    #                   300, 0.1, 0.1,
    #                   0.1,
    #                   50,
    #                   2000, 10, 1)
    # 下面是使用embedding的方法
    # get_tokenizer_and_train("../build/divided_code", "../build/tokenizer", "../build/treated_str_train", 5, 300)
    get_trained_model_using_embed("../build/tokenizer", "../test/test", "../build/treated_str_test", 300)
    # get_trained_model("../build/treated_str_train", "../build/model", "../build/tokenizer",
    #                   300,
    #                   50,
    #                   50, 0.1, 0.1,
    #                   0.1,
    #                   50,
    #                   1, 0.1, 1)
