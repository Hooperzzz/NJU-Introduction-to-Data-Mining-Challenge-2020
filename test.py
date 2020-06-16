import pickle
import GenerateAns
from Debug import Assert
from executeCode import ExeResult


def judge(res1: ExeResult, res2: ExeResult) -> bool:
    if res1.input_times != res2.input_times:
        return False
    if len(res1.output.split()) != len(res2.output.split()):
        return False
    for str1, str2 in zip(res1.output.split(), res2.output.split()):
        d1 = str1.replace(".", "").isdigit()
        d2 = str2.replace(".", "").isdigit()
        if not d1 and not d2:
            # 两个都不是数字
            s1 = str1.replace("\n", "").replace("\r", "").replace(" ", "").replace("\t", "")
            s2 = str2.replace("\n", "").replace("\r", "").replace(" ", "").replace("\t", "")
            if s1 != s2:
                return False
        elif d1 and d2:
            if ("." in res1.output and "." in res2.output) or ("." not in res1.output and "." not in res2.output):
                if int(str1.replace(".", "")) != int(str2.replace(".", "")):
                    return False
            else:
                return False
        else:
            return False
    return True


def compare_test(list1: list, list2: list) -> bool:
    Assert(len(list1) == len(list2), "两组测试的结果数目不一致")
    for res1, res2 in zip(list1, list2):
        if not judge(res1, res2):
            return False
    return True


if __name__ == "__main__":
    with open("build/save", "rb") as f:
        exe_res = pickle.load(f)

    g = GenerateAns.GenerateAns("ans.csvc", "sample_submission.csv")
    while True:
        t = g.get()
        if t is None:
            break
        g.add(1 if compare_test(exe_res[t[0]], exe_res[t[1]]) else 0)
