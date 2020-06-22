import pickle
from GenerateAns import GenerateAns
from compareTestCsv import compare_test2 as compare_test
from StaticCodeAnalysisFunc import analysis_str, analysis_input_output_loop_num
from network_part.PredictByNetwork import PredictByNetwork


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
        model_path: str, word2vec_model_path: str, test_dir_path: str, test_len: int, word_len: int, thresh_num: float
        ) -> None:
    """
    辅助以lstm
    """
    with open(save_path, "rb") as f:
        exe_res = pickle.load(f)
    predict_by_network = PredictByNetwork(model_path, word2vec_model_path, test_dir_path,
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


if __name__ == "__main__":
    # use_exe_save("save/execute_info", "new_ans.csv")
    # use_exe_save_and_analysis_str("save/execute_info", "new_ans.csv")
    # use_exe_save_and_analysis_str_input_output_loop("save/execute_info", "new_ans.csv")
    use_exe_save_and_analysis_str_and_network("save/execute_info", "new_ans.csv",
                                              "save/model", "save/word2vec_model", "save/test_vectoried_test",
                                              1000, 50, 0.5)
