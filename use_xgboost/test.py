import os
import random
import pickle

from Debug import Log
from use_xgboost.GetCompareResult import GetCompareResult
import xgboost as xgb
from sklearn.metrics import f1_score
import sklearn
import pandas as pd


def get_trained_xgb(train_path: str, save_path: str, prec: float):
    get_what = []
    for i in range(83):
        for j in range(int(500 * prec)):
            get_what.append(i * 500 + j)
    Log("开始训练模型")
    for i in range(83):
        csv = pd.read_csv(os.path.join(train_path, str(i) + ".csv"))
        train = csv.iloc[get_what, 1:].values
        labels = csv.iloc[get_what, :1].values
        dtrain = xgb.DMatrix(train, label=labels)
        param = {'max_depth': 16, 'nthread': 4, 'gamma': 0.00001, 'objective': 'binary:logistic'}
        bst = xgb.train(param, dtrain, 16)
        bst.save_model(os.path.join(save_path, str(i) + "_report.model"))

        train_preds = bst.predict(dtrain)
        train_predictions = [round(value) for value in train_preds]
        y_train = dtrain.get_label()  # 值为输入数据的第一行
        train_accuracy = f1_score(y_train, train_predictions)
        Log("Train f1: %.2f%%" % (train_accuracy * 100.0))

        # from matplotlib import pyplot
        # import graphviz
        # xgb.plot_tree(bst, num_trees=0, rankdir='LR')
        # pyplot.show()


def get_predict(model_path: str, test_path: str, save_path: str, prec: float):
    get_what = []
    for i in range(83):
        for j in range(int(500 * prec), 500):
            get_what.append(i * 500 + j)
    model = []
    for i in range(83):
        model.append(xgb.Booster(model_file=os.path.join(model_path, str(i) + "_report.model")))
    csv = pd.read_csv(test_path, header=None)
    train = csv.iloc[get_what, 1:].values
    labels = csv.iloc[get_what, :1].values
    dtest = xgb.DMatrix(train, label=labels)
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


def get_random_train_result(train_all_result_path: str, test_num: int, is_clone_test: bool,
                            test_index_map: str, network_result_path: str, prec: float):
    with open(train_all_result_path, "rb") as f:
        train_all_result = pickle.load(f)

    gcr = GetCompareResult(test_index_map, network_result_path)

    clone_num = 0
    length = 500 - int(500 * prec)
    for _ in range(test_num):
        if is_clone_test:
            i = random.randint(0, 83 - 1)
            j1 = random.randint(0, 500 - 1 - int(500 * prec))
            j2 = j1
            while j1 == j2:
                j2 = random.randint(0, 500 - 1 - int(500 * prec))
            loc1 = i * length + j1
            loc2 = i * length + j2
        else:
            i1 = random.randint(0, 83 - 1)
            i2 = i1
            while i1 == i2:
                i2 = random.randint(0, 83 - 1)
            j1 = random.randint(0, 500 - 1 - int(500 * prec))
            j2 = random.randint(0, 500 - 1 - int(500 * prec))
            loc1 = i1 * length + j1
            loc2 = i2 * length + j2

        if gcr.compare_method(train_all_result[loc1], train_all_result[loc2]):
            clone_num += 1

    return clone_num


if __name__ == "__main__":
    # get_trained_xgb("..\\build", "..\\build", 0.66)
    get_predict("..\\build", "..\\build\\0.csv", "../build/network_result_for_report", 0.66)
    get_random_train_result("../build/network_result_for_report", 10000, False,
                            "../build/test_index_map", "../build/network_result", 0.66)
