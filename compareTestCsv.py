import pickle
import GenerateAns
from Debug import Assert
from executeCode import ExeResult


def compare_test1(list1: list, list2: list) -> bool:
    # 简单版
    def judge(res_1: ExeResult, res_2: ExeResult) -> bool:
        return res_1 == res_2

    Assert(len(list1) == len(list2), "两组测试的结果数目不一致")
    for res1, res2 in zip(list1, list2):
        if not judge(res1, res2):
            return False
    return True


def judge2(res_1: ExeResult, res_2: ExeResult) -> bool:
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


def compare_test2(list1: list, list2: list) -> bool:

    Assert(len(list1) == len(list2), "两组测试的结果数目不一致")
    for res1, res2 in zip(list1, list2):
        if not judge2(res1, res2):
            return False
    return True


def compare_test3(list1: list, list2: list) -> bool:
    # 在2的基础上改不考虑输入数量是否相同（效果不佳，会多出很多的1）
    def judge(res_1: ExeResult, res_2: ExeResult) -> bool:
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


def compare_test4(list1: list, list2: list) -> bool:
    # 在2的基础上改，这次只要有一组输出正确且不是编译错误或运行错误，就会输出True(效果变差，0.69)
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
        if res1.result == 0 and res2.result == 0 and judge(res1, res2):
            return True
    return False


def compare_test5(list1: list, list2: list) -> bool:
    # 在2的基础上改，这次只要有一组输出正确且不是编译错误(不含运行错误)，就会输出True(效果变差,0.70)
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
        if res1.result != -1 and res2.result != -1 and judge(res1, res2):
            return True
    return False


def compare_test6(list1: list, list2: list) -> bool:
    # 在judge中跳过所有的ret为-1和139的样例，需要至少有一组相同才输出为True
    # 提升到了0.76（与xgboost的组合下）
    Assert(len(list1) == len(list2), "两组测试的结果数目不一致")
    is_have_at_least_one_same = False
    for res1, res2 in zip(list1, list2):
        if res1.result == -1 or res2.result == -1 or res1.result == 139 or res2.result == 139:
            continue
        if not judge2(res1, res2):
            return False
        is_have_at_least_one_same = True
    if is_have_at_least_one_same:
        ret = True
    else:
        ret = False
    return ret


def compare_test7(list1: list, list2: list) -> int:
    # 在judge中跳过所有的ret为-1和139的样例，需要至少有一组相同才输出为True
    Assert(len(list1) == len(list2), "两组测试的结果数目不一致")
    is_have_at_least_one_same = 0
    for res1, res2 in zip(list1, list2):
        if res1.result == -1 or res2.result == -1 or res1.result == 139 or res2.result == 139:
            continue
        if not judge2(res1, res2):
            return False
        is_have_at_least_one_same += 1
    return is_have_at_least_one_same


if __name__ == "__main__":
    with open("save/execute_info", "rb") as f:
        exe_res = pickle.load(f)

    g = GenerateAns.GenerateAns("ans.csvc", "sample_submission.csv")
    while True:
        t = g.get()
        if t is None:
            break
        g.add(1 if compare_test2(exe_res[t[0]], exe_res[t[1]]) else 0)
