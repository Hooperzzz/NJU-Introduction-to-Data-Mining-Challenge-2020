import random

from executeCode import ExecuteCode
import os
import PathFunc
from Debug import Log, GLog
import GenerateAns
import pickle
import time
from compareTestCsv import compare_test6 as compare_test
from GenerateAns import GenerateAns


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


def test_test(path: str, is_save_intermediate_file: bool = False) -> None:
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

    with open("../build/execute_info", "wb") as f:
        pickle.dump(exe_res, f)

    with open("../build/execute_info", "rb") as f:
        exe_res = pickle.load(f)

    g = GenerateAns.GenerateAns("../ans.csv", "../sample_submission.csv")
    while True:
        t = g.get()
        if t is None:
            break
        g.add(1 if compare_test(exe_res[t[0]], exe_res[t[1]]) else 0)


def test_test_thread(path: str, thread_num: int, is_save_intermediate_file: bool = False) -> None:
    # 有bug，废弃
    Log(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    Log("开始测试", path, "的数据")
    exe = ExecuteCode(is_save_intermediate_file=is_save_intermediate_file)
    exe_res, compile_succ = [], 0
    for root, dirs, files in os.walk(path):
        exe_res, compile_succ = exe.execute_code_thread(PathFunc.to_linux(path), files, thread_num)
        break
    Log("编译成功数量：", compile_succ, "/", len(files))

    with open("build/execute_info", "wb") as f:
        pickle.dump(exe_res, f)

    g = GenerateAns.GenerateAns("ans.csvc", "sample_submission.csv")
    while True:
        t = g.get()
        if t is None:
            break
        g.add(1 if compare_test(exe_res[t[0]], exe_res[t[1]]) else 0)

    Log(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


def test_train_random_from_same_dir(dir_path: str, test_num: int, is_save_intermediate_file: bool = False):
    dir_list = os.listdir(dir_path)
    dir_list_len = len(dir_list)
    dir_file = []
    for name in dir_list:
        dir_file_list = os.listdir(os.path.join(dir_path, name))
        dir_file.append([name, len(dir_file_list), dir_file_list])

    exe = ExecuteCode(is_save_intermediate_file=is_save_intermediate_file)
    is_same_num, succ_num = 0, 0
    for _ in range(test_num):
        i = random.randint(0, dir_list_len-1)
        j1 = random.randint(0, dir_file[i][1]-1)
        j2 = random.randint(0, dir_file[i][1]-1)
        Log("开始比较 ", dir_file[i][0], " 的 ", dir_file[i][2][j1], " 和 ", dir_file[i][2][j2])
        p1 = PathFunc.join(dir_path, dir_file[i][0], dir_file[i][2][j1])
        p2 = PathFunc.join(dir_path, dir_file[i][0], dir_file[i][2][j2])
        ret_list1, _ = exe.execute_code(p1)
        ret_list2, _ = exe.execute_code(p2)
        if not compare_test(ret_list1, ret_list2):
            GLog("文件不相同")
            for r1, r2 in zip(ret_list1, ret_list2):
                GLog(r1, r2)
            if sum([1 if x.result == 0 else 0 for x in ret_list1]) != len(ret_list1) \
                or sum([1 if x.result == 0 else 0 for x in ret_list2]) != len(ret_list2):
                print("diff")
            else:
                succ_num += 1
        else:
            is_same_num += 1
            succ_num += 1
    return is_same_num, succ_num


def test_train_random_from_different_dir(dir_path: str, test_num: int, is_save_intermediate_file: bool = False):
    dir_list = os.listdir(dir_path)
    dir_list_len = len(dir_list)
    dir_file = []
    for name in dir_list:
        dir_file_list = os.listdir(os.path.join(dir_path, name))
        dir_file.append([name, len(dir_file_list), dir_file_list])

    exe = ExecuteCode(is_save_intermediate_file=is_save_intermediate_file)
    is_different_num, succ_num = 0, 0
    for _ in range(test_num):
        i1 = random.randint(0, dir_list_len-1)
        j1 = random.randint(0, dir_file[i1][1]-1)
        i2 = random.randint(0, dir_list_len-1)
        j2 = random.randint(0, dir_file[i2][1]-1)
        if i1 == i2:
            continue
        Log("开始比较 ", dir_file[i1][0], " 的 ", dir_file[i1][2][j1], " 和 ", dir_file[i2][0], " 的 ", dir_file[i2][2][j2])
        p1 = PathFunc.join(dir_path, dir_file[i1][0], dir_file[i1][2][j1])
        p2 = PathFunc.join(dir_path, dir_file[i2][0], dir_file[i2][2][j2])
        ret_list1, _ = exe.execute_code(p1)
        ret_list2, _ = exe.execute_code(p2)
        if compare_test(ret_list1, ret_list2):
            GLog("文件相同")
            for r1, r2 in zip(ret_list1, ret_list2):
                GLog(r1, r2)
        else:
            is_different_num += 1
            if sum([1 if x.result == 0 else 0 for x in ret_list1]) != len(ret_list1) \
                or sum([1 if x.result == 0 else 0 for x in ret_list2]) != len(ret_list2):
                print("diff")
            else:
                succ_num += 1
    print(is_different_num, "   ", succ_num)
    return is_different_num, succ_num


def get_num_of_succ_and_compilefail_and_runningfail(save_path: str):
    # 统计一下成功，编译失败，运行失败的数量，用于写报告
    with open(save_path, "rb") as f:
        exe_res = pickle.load(f)

    total_succ, part_succ, total_cf, part_cf, total_rf, part_rf = 0, 0, 0, 0, 0, 0
    for res in exe_res.values():
        length = len(res)
        succ_num = sum([1 if x.result == 0 else 0 for x in res])
        cf_num = sum([1 if x.result != -1 else 0 for x in res])
        rf_num = sum([1 if x.result != -1 else 0 for x in res])

        if succ_num == length:
            total_succ += 1
        elif succ_num > 0:
            part_succ += 1

        if cf_num == length:
            total_cf += 1
        elif cf_num > 0:
            part_cf += 1

        if rf_num == length:
            total_rf += 1
        elif rf_num > 0:
            part_rf += 1

    return total_succ, part_succ, total_cf, part_cf, total_rf, part_rf



if __name__ == "__main__":
    # test_train_in_one_dir(PathFunc.to_windows(".\\train\\train\\0ae1"), 30, True)
    # test_train(PathFunc.to_windows(".\\train\\train"), 6, True)
    # test_test("./test/test", True)
    # test_test_thread("test/test", 4, True) # 有bug，废弃
    test_train_random_from_same_dir("./train/train", 100, False)  # 用于写报告的
    # test_train_random_from_different_dir("./train/train", 100, False)  # 用于写报告的
    # get_num_of_succ_and_compilefail_and_runningfail("./build/execute_info")  # 用于写报告的
