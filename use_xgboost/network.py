import os
import pickle
import re

from tensorflow.keras.preprocessing import text
import xgboost as xgb
from sklearn.metrics import f1_score

from Debug import Log


def discrete_code(code: str) -> list:
    # 对代码进行分词
    all_word = []
    word = re.split(
        "!|%|\\^|&|\\*|\\(|\\)|-|\\+|=|\\{|\\}|\[|\]|:|;|<|,|>|\\.|\\?|/|\n|\t|\r|'|\"| ",
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
    return word


def deal_with_s(word_map: dict, s: list, word_num: int):
    test = [0 for _ in range(word_num)]
    for w in s:
        if w in word_map.keys():
            test[word_map[w]] += 1
        else:
            test[word_num - 1] += 1
    return test


def get_divided_code(train_path: str, map_save_path: str, train_save_path: str, save_tokenizer_path: str,
                     word_num: int) -> None:
    # 在代码中不同成分之间添加空格
    all_word = []
    all_word_in_one_line = []
    for root, dirs, files in os.walk(train_path):
        dir_test = []
        for name in files:
            with open(os.path.join(root, name), "r") as f:
                code = f.read()
            r = discrete_code(code)
            dir_test.append(r)
            all_word_in_one_line.extend(r)
        if dir_test:
            all_word.append(dir_test)
    tokenizer = text.Tokenizer(lower=False, char_level=False)
    tokenizer.fit_on_texts([all_word_in_one_line])
    with open(save_tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)
    selected_word = sorted(tokenizer.word_counts.items(), key=lambda a: -a[1])
    selected_word = selected_word[:word_num - 1]
    word_map = dict()
    count = 0
    for w, _ in selected_word:
        word_map[w] = count
        count += 1
    with open(map_save_path, "wb") as f:
        pickle.dump(word_map, f)
    test_count = []
    for d in all_word:
        dir_test = []
        for s in d:
            dir_test.append(deal_with_s(word_map, s, word_num))
        test_count.append(dir_test)
    with open(train_save_path, "wb") as f:
        pickle.dump(test_count, f)


def get_divided_test(map_path: str, test_dir_path: str, save_path: str, word_num: int):
    with open(map_path, "rb") as f:
        word_map = pickle.load(f)
    all_word_dict = dict()
    for root, dirs, files in os.walk(test_dir_path):
        for name in files:
            with open(os.path.join(root, name), "r") as f:
                code = f.read()
            dc = discrete_code(code)
            add_sentence = deal_with_s(word_map, dc, word_num)
            if root not in all_word_dict.keys():
                all_word_dict[root] = dict()
            all_word_dict[root][name.split(".")[0]] = add_sentence
    with open(save_path, "wb") as f:
        pickle.dump(all_word_dict, f)


def get_train_csv_file(save_path: str, train_treated_path: str):
    with open(train_treated_path, "rb") as f:
        train_treated = pickle.load(f)
    len1 = len(train_treated)
    for i in range(len1):
        with open(os.path.join(save_path, str(i) + ".csv"), "w") as f:
            for j in range(len1):
                for s in train_treated[j]:
                    f.write(str(1 if i == j else 0))
                    for w in s:
                        f.write("," + str(w))
                    f.write("\n")


def get_test_csv_file(save_path: str, test_treated_path: str):
    with open(test_treated_path, "rb") as f:
        test_treated = pickle.load(f)
    with open(os.path.join(save_path, "test.csv"), "w") as f:
        for _, s in test_treated["../test/test"].items():
            f.write(str(0))
            for w in s:
                f.write("," + str(w))
            f.write("\n")


def get_test_index_map(test_treated_path: str, save_path: str) -> None:
    with open(test_treated_path, "rb") as f:
        test_treated = pickle.load(f)
    test_index_map = dict()
    count = 0
    for name, _ in test_treated["../test/test"].items():
        test_index_map[name] = count
        count += 1
    with open(save_path, "wb") as f:
        pickle.dump(test_index_map, f)


def get_trained_xgb(train_path: str, save_path: str):
    Log("开始训练模型")
    for i in range(83):
        dtrain = xgb.DMatrix(os.path.join(train_path, str(i) + ".csv") + "?format=csv&label_column=0")
        param = {'max_depth': 16, 'nthread': 4, 'gamma': 0.00001, 'objective': 'binary:logistic'}
        bst = xgb.train(param, dtrain, 16)
        bst.save_model(os.path.join(save_path, str(i) + ".model"))

        train_preds = bst.predict(dtrain)
        train_predictions = [round(value) for value in train_preds]
        y_train = dtrain.get_label()  # 值为输入数据的第一行
        train_accuracy = f1_score(y_train, train_predictions)
        Log("Train f1: %.2f%%" % (train_accuracy * 100.0))

        # from matplotlib import pyplot
        # import graphviz
        # xgb.plot_tree(bst, num_trees=0, rankdir='LR')
        # pyplot.show()


def get_predict(model_path: str, test_path: str, save_path: str):
    model = []
    for i in range(83):
        model.append(xgb.Booster(model_file=os.path.join(model_path, str(i) + ".model")))
    dtest = xgb.DMatrix(test_path + "?format=csv&label_column=0")
    p = []
    for m in model:
        p.append(m.predict(dtest))

    trans_p = []
    len1, len2 = len(p), len(p[0])
    for i in range(len2):
        t = []
        for j in range(len1):
            t.append(p[j][i])
        trans_p.append(t)

    with open(save_path, "wb") as f:
        pickle.dump(trans_p, f)


if __name__ == "__main__":
    # get_divided_code('../train/train', "../build/map_save", "../build/train_teated", "../build/tokenizer", 300)
    # get_divided_test("../build/map_save", "../test/test", "../build/test_teated", 300)
    # get_train_csv_file("../build", "../build/train_teated")
    # get_test_csv_file("../build", "../build/test_teated")
    # get_test_index_map("../build/test_teated", "../build/test_index_map")
    # get_trained_xgb("..\\build", "..\\build")
    get_predict("..\\build", "..\\build\\test.csv", "../build/network_result")
