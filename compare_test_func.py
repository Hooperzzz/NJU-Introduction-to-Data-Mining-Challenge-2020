import pickle
import GenerateAns
from Debug import Assert
from executeCode import ExeResult


def compare_test1(list1: list, list2: list) -> bool:
    def judge(res_1: ExeResult, res_2: ExeResult) -> bool:
        return res_1 == res_2

    Assert(len(list1) == len(list2), "两组测试的结果数目不一致")
    for res1, res2 in zip(list1, list2):
        if not judge(res1, res2):
            return False
    return True


def compare_test2(list1: list, list2: list) -> bool:
    def judge(res_1: ExeResult, res_2: ExeResult) -> bool:
        if res_1.input_times != res_2.input_times:
            return False
        if len(res_1.output.split()) != len(res_2.output.split()):
            return False
        for str1, str2 in zip(res_1.output.split(), res_2.output.split()):
            d1 = str1.replace(".", "").isdigit()
            d2 = str2.replace(".", "").isdigit()
            if not d1 and not d2:
                # 两个都不是数字
                s1 = str1.replace("\n", "").replace("\r", "").replace(" ", "").replace("\t", "")
                s2 = str2.replace("\n", "").replace("\r", "").replace(" ", "").replace("\t", "")
                if s1 != s2:
                    return False
            elif d1 and d2:
                if ("." in res_1.output and "." in res_2.output) or (
                        "." not in res_1.output and "." not in res_2.output):
                    if int(str1.replace(".", "")) != int(str2.replace(".", "")):
                        return False
                else:
                    return False
            else:
                return False
        return True

    Assert(len(list1) == len(list2), "两组测试的结果数目不一致")
    for res1, res2 in zip(list1, list2):
        if not judge(res1, res2):
            return False
    return True


if __name__ == "__main__":
    with open("save/save", "rb") as f:
        exe_res = pickle.load(f)

    g = GenerateAns.GenerateAns("ans.csvc", "sample_submission.csv")
    while True:
        t = g.get()
        if t is None:
            break
        g.add(1 if compare_test2(exe_res[t[0]], exe_res[t[1]]) else 0)
