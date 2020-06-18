import pickle
from GenerateAns import GenerateAns
from compareTestCsv import compare_test2 as compare_test
from executeCode import ExecuteCode, ExeResult
from StaticCodeAnalysisFunc import analysis_str


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


if __name__ == "__main__":
    # use_exe_save("save/save", "new_ans.csv")
    use_exe_save_and_analysis_str("save/save", "new_ans.csv")
