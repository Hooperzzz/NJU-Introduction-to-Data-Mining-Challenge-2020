import pickle
from GenerateAns import GenerateAns
from compare_test_func import compare_test2 as compare_test


def use_exe_save(save_path: str, csv_name: str) -> None:
    with open(save_path, "rb") as f:
        exe_res = pickle.load(f)

    g = GenerateAns.GenerateAns(csv_name, "sample_submission.csv")
    while True:
        t = g.get()
        if t is None:
            break
        g.add(1 if compare_test(exe_res[t[0]], exe_res[t[1]]) else 0)


if __name__ == "__main__":
    use_exe_save("save/save", "new_ans.csv")
