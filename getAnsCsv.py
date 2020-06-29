import pickle

from Debug import Log
from GenerateAns import GenerateAns
from compareTestCsv import compare_test6 as compare_test
from compareTestCsv import compare_test7
from compareTestCsv import judge2 as judge
from StaticCodeAnalysisFunc import analysis_str, analysis_input_output_loop_num
from network_part.PredictByNetwork import PredictByNetwork
import network_part.UseNetwork as UseNetwork
from use_xgboost.GetCompareResult import GetCompareResult


def use_exe_save(save_path: str, csv_name: str) -> None:
    with open(save_path, "rb") as f:
        exe_res = pickle.load(f)

    g = GenerateAns(csv_name, "sample_submission.csv")
    while True:
        t = g.get()
        if t is None:
            break
        g.add(1 if compare_test(exe_res[t[0]], exe_res[t[1]]) else 0)


def use_exe_save_and_analysis_str(save_path: str, csv_name: str) -> None:
    """
    有点点效果，提升到了0.74
    """
    with open(save_path, "rb") as f:
        exe_res = pickle.load(f)

    g = GenerateAns(csv_name, "sample_submission.csv")
    while True:
        t = g.get()
        if t is None:
            break
        g.add(1 if compare_test(exe_res[t[0]], exe_res[t[1]]) or analysis_str(t[0], t[1]) else 0)


def use_exe_save_and_analysis_str_input_output_loop(save_path: str, csv_name: str) -> None:
    """
    效果不佳，调出的最好成绩为0.70
    """
    with open(save_path, "rb") as f:
        exe_res = pickle.load(f)

    g = GenerateAns(csv_name, "sample_submission.csv")
    while True:
        t = g.get()
        if t is None:
            break
        g.add(1 if compare_test(exe_res[t[0]], exe_res[t[1]])
                   or analysis_str(t[0], t[1])
                   or analysis_input_output_loop_num(t[0], t[1])
              else 0)


def use_exe_save_and_analysis_str_and_network(
        save_path: str, csv_name: str,
        model_path: str, test_dir_path: str,
        test_len: int, word_len: int, thresh_num: float
) -> None:
    """
    辅助以lstm
    """
    with open(save_path, "rb") as f:
        exe_res = pickle.load(f)
    predict_by_network = PredictByNetwork(model_path, test_dir_path,
                                          test_len, word_len, thresh_num)
    g = GenerateAns(csv_name, "sample_submission.csv")
    while True:
        t = g.get()
        if t is None:
            break
        g.add(1 if compare_test(exe_res[t[0]], exe_res[t[1]])
                   or analysis_str(t[0], t[1])
                   or predict_by_network.predict(t[0], t[1])
              else 0)


def use_exe_save_and_analysis_str_and_csv(
        save_path: str, compared_csv_path: str, csv_name: str,
        model_path: str, test_dir_path: str,
        test_len: int, word_len: int, thresh_num: float
) -> None:
    """
    用lstm生成的答案的csv文件来生成
    """
    with open(save_path, "rb") as f:
        exe_res = pickle.load(f)
    predict_by_network = PredictByNetwork(model_path, test_dir_path,
                                          test_len, word_len, thresh_num)
    g = GenerateAns(csv_name, "sample_submission.csv")
    compared_csv = GenerateAns("", compared_csv_path)
    while True:
        t = g.get()
        compared_csv.get()
        if t is None:
            break
        b1 = compare_test(exe_res[t[0]], exe_res[t[1]])
        b2 = analysis_str(t[0], t[1])
        b3 = ((judge(exe_res[t[0]][0], exe_res[t[1]][0]) and exe_res[t[0]][0].result != -1)
               or (judge(exe_res[t[0]][1], exe_res[t[1]][1]) and exe_res[t[1]][0].result != -1))\
             and compared_csv.get_cur_value() == 1
        p = 1 if b1 or b2 or b3 else 0
        if not b1 and not b2:
            print("hehehe")
        g.add(p)


def use_xgboost(test_index_map: str, network_result_path: str, csv_name: str) -> None:
    gcr = GetCompareResult(test_index_map, network_result_path)
    g = GenerateAns(csv_name, "sample_submission.csv")
    while True:
        t = g.get()
        if t is None:
            break
        g.add(1 if gcr.get(t[0], t[1]) else 0)


def use_exe_save_and_analysis_str_and_xgboost(save_path: str, csv_name: str,
                                              test_index_map: str, network_result_path: str) -> None:
    with open(save_path, "rb") as f:
        exe_res = pickle.load(f)
    gcr = GetCompareResult(test_index_map, network_result_path)

    g = GenerateAns(csv_name, "sample_submission.csv")
    while True:
        t = g.get()
        if t is None:
            break
        xgboost_bool = gcr.get(t[0], t[1])
        # xgboost_bool = False
        g.add(1 if compare_test(exe_res[t[0]], exe_res[t[1]]) or analysis_str(t[0], t[1]) or xgboost_bool else 0)


def use_exe_save_and_analysis_str_and_xgboost_strict(save_path: str, csv_name: str,
                                              test_index_map: str, network_result_path: str) -> None:
    # 判断更为严格：只有在运行的时候有运行失败的情况才进行后面的补充判断
    with open(save_path, "rb") as f:
        exe_res = pickle.load(f)
    gcr = GetCompareResult(test_index_map, network_result_path)

    g = GenerateAns(csv_name, "./build/sample_submission.csv")
    while True:
        t = g.get()
        if t is None:
            break
        xgboost_bool = gcr.get(t[0], t[1])
        min_succ_num = min(sum([1 if x.result == 0 else 0 for x in exe_res[t[0]]]),
                           sum([1 if x.result == 0 else 0 for x in exe_res[t[1]]]))
        # xgboost_bool = False
        same_num = compare_test7(exe_res[t[0]], exe_res[t[1]])
        g.add(1 if
              same_num == len(exe_res[t[0]]) or analysis_str(t[0], t[1])
              or ((min_succ_num == 0 or same_num >= 1) and xgboost_bool)
              else 0)


if __name__ == "__main__":
    # use_exe_save("save/execute_info", "new_ans.csv")
    # use_exe_save_and_analysis_str("save/execute_info", "new_ans.csv")
    # use_exe_save_and_analysis_str_input_output_loop("save/execute_info", "new_ans.csv")
    # use_exe_save_and_analysis_str_and_network("save/execute_info", "new_ans.csv",
    #                                           "build/model", "build/test_vectoried_test",
    #                                           500, 20, 0.90)
    # use_exe_save_and_analysis_str_and_network("save/execute_info", "new_ans.csv",
    #                                           "build/model", "build/treated_str_test",
    #                                           300, 20, 0.90) # test运行到了的位置79011
    # use_exe_save_and_analysis_str_and_csv("save/execute_info", "new_ans1.csv", "new_ans.csv",
    #                                       "build/model", "build/treated_str_test",
    #                                       300, 20, 0.90)
    # use_xgboost("./build/test_index_map", "./build/network_result", "new_ans.csv")
    use_exe_save_and_analysis_str_and_xgboost("build/execute_info", "new_ans.csv",
                                              "./build/test_index_map", "./build/network_result")
