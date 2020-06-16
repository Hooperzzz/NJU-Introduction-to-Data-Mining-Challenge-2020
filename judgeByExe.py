from executeCode import ExecuteCode, ExeResult
import os
import PathFunc
from Debug import Log, Assert
import GenerateAns
import pickle
import time


def judge(res1: ExeResult, res2: ExeResult) -> bool:
    return res1 == res2


def compare_test(list1: list, list2: list) -> bool:
    Assert(len(list1) == len(list2), "两组测试的结果数目不一致")
    for res1, res2 in zip(list1, list2):
        if not judge(res1, res2):
            return False
    return True


def test_train_in_one_dir(path: str, test_num: int = -1, is_save_intermediate_file=False) -> None:
    exe_res = []
    exe = ExecuteCode(is_save_intermediate_file=is_save_intermediate_file)
    Log("开始测试", path, "的数据")
    count = 0
    for root, dirs, files in os.walk(path):
        for name in files:
            if count == test_num:
                break
            count += 1
            Log("开始编译", name)
            p = PathFunc.join(PathFunc.to_linux(path), name)
            exe_res.append({"res": exe.execute_code(p), "name": name})
        break

    Log("开始比较输出")
    length = len(exe_res)
    for i in range(length):
        for j in range(i+1, length):
            if not compare_test(exe_res[i]["res"], exe_res[j]["res"]):
                l1, l2 = exe_res[i], exe_res[j]
                print("文件", exe_res[i]["name"], "和", exe_res[j]["name"], "结果不一致")
                continue


def test_train(path: str, test_num_in_one_dir: int = -1, is_save_intermediate_file=False) -> None:
    for root, dirs, files in os.walk(path):
        for name in dirs:
            p = os.path.join(path, name)
            test_train_in_one_dir(p, test_num_in_one_dir, is_save_intermediate_file)
        break


def test_test(path: str, is_save_intermediate_file: bool = False):
    Log("开始测试", path, "的数据")
    exe_res, compile_succ = dict(), 0
    exe = ExecuteCode(is_save_intermediate_file=is_save_intermediate_file)
    for root, dirs, files in os.walk(path):
        total_num, cur_num = len(files), 1
        for name in files:
            Log("开始编译", name, "：[", str(cur_num), "/", str(total_num), "]")
            cur_num += 1
            p = PathFunc.join(PathFunc.to_linux(path), name)
            exe_res[name.split(".")[0]], cs = exe.execute_code(p)
            compile_succ += cs
        break
    Log("编译成功数量：", compile_succ, "/", len(files))

    with open("build/save", "wb") as f:
        pickle.dump(exe_res, f)

    g = GenerateAns.GenerateAns("ans.csv", "sample_submission.csv")
    while True:
        t = g.get()
        if t is None:
            break
        g.add(1 if compare_test(exe_res[t[0]], exe_res[t[1]]) else 0)


def test_test_thread(path: str, thread_num: int, is_save_intermediate_file: bool = False):
    Log(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    Log("开始测试", path, "的数据")
    exe = ExecuteCode(is_save_intermediate_file=is_save_intermediate_file)
    exe_res, compile_succ = [], 0
    for root, dirs, files in os.walk(path):
        exe_res, compile_succ = exe.execute_code_thread(PathFunc.to_linux(path), files, thread_num)
        break
    Log("编译成功数量：", compile_succ, "/", len(files))

    with open("build/save", "wb") as f:
        pickle.dump(exe_res, f)

    g = GenerateAns.GenerateAns("ans.csvc", "sample_submission.csv")
    while True:
        t = g.get()
        if t is None:
            break
        g.add(1 if compare_test(exe_res[t[0]], exe_res[t[1]]) else 0)

    Log(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


if __name__ == "__main__":
    # test_train_in_one_dir(PathFunc.to_windows("train\\train\\0ae1"), 30, True)
    # test_train(PathFunc.to_windows("train\\train"), 6, True)
    test_test("test/test", True)
    # test_test_thread("test/test", 4, True)
