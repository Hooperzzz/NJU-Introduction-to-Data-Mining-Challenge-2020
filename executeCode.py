"""
为测试加上必要的，代码头部，并替换输入函数以统计输入次数
以input_data作为输入，获取输出，输入次数以及代码执行结果
"""
import subprocess
import traceback
import generateTest
import PathFunc
from Debug import Log, Assert, Warn
import re
import threading


class ExeResult:
    def __init__(self, output: str, input_times: int, result: int):
        self.output = output
        self.input_times = input_times
        self.result = result

    def __eq__(self, other):
        return \
            self.result != 0 or other.result != 0 or \
            (self.output == other.output and
             self.input_times == other.input_times and
             self.result == other.result)


class ExecuteCode:
    work_path = "build"
    intermediate_file_dir = work_path

    code_file_name = "test"
    exe_file_name = "test"
    code_file_front = "#include\"../AddHead.h\"\r\n"

    input_data = [
        generateTest.get_repetition_test(1, 100),
        generateTest.get_repetition_test(3, 100)
    ]

    def __init__(self, is_save_intermediate_file: bool = False):
        self.is_save_intermediate_file = is_save_intermediate_file
        self.pattern = re.compile(r'VAR[0-9a-zA-Z_]*')

    def pretreat_code(self, new_code: str) -> str:
        """
        添加头部，以及对代码进行一些修改，尽量使其可以运行
        目前碰到过的，导致编译不通过的问题：
            scanf对long使用%d(把代码中的所有整数类型都变成int)
            使用long int这种比较少见的类型
            使用gets函数，但目前编译器不太支持了（替换成cin.getline）
            有未定义的变量（估计是由于宏定义被去掉了）
        """
        var_list_str = ""
        """
        var_list = set(self.pattern.findall(new_code))
        for s in var_list:
            var_list_str += "const int " + s + " = 100;\n"
        var_list_str += "\n"
        """

        new_code = self.code_file_front + var_list_str + new_code
        new_code = new_code.replace("void main()", "int main()")
        new_code = new_code.replace("void main(void)", "int main()")  # test/de362e91af0447e8
        new_code = new_code.replace("unsigned", "")
        new_code = new_code.replace("short", "int")
        new_code = new_code.replace("long", "int")
        new_code = new_code.replace("int int", "int")
        new_code = new_code.replace("%hd", "%d")
        new_code = new_code.replace("%ld", "%d")
        new_code = new_code.replace("%ld", "%d")
        new_code = new_code.replace("gets", "cin.getline")
        return new_code

    def execute_code(self, test_path: str, special_mask: str = "") -> [list, int]:
        # 首先把测试文件变成可执行的代码
        with open(test_path, 'r') as f:
            new_code = f.read()
        new_code = self.pretreat_code(new_code)
        # 输出代码文件
        test_name = PathFunc.split(test_path)[-1]
        if self.is_save_intermediate_file:
            code_path = PathFunc.join(self.intermediate_file_dir, test_name + '.cc')
        else:
            code_path = PathFunc.join(self.work_path, self.code_file_name + special_mask + ".cc")
        with open(code_path, 'w') as f:
            f.write(new_code)
        # 编译出可执行文件
        exe_path = PathFunc.join(self.work_path, self.exe_file_name + special_mask)
        compile_result = subprocess.run(["wsl.exe", "g++", code_path, "-o", exe_path, "-w"],
                                        stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE)
        if compile_result.returncode != 0:
            compile_succ = 0
            Warn(False, special_mask, "编译失败：", test_path, special_mask=special_mask)
        else:
            compile_succ = 1
        # 运行代码，记录输入以及输出
        code_return = []
        for t in self.input_data:
            process = subprocess.Popen(["wsl.exe", exe_path],
                                       stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
            ret, out = 0, None
            try:
                exe_stdout, exe_stderr = process.communicate(input=t, timeout=5)
                ret = process.poll()
                input_times = 0 if exe_stderr.decode() == "" else int(exe_stderr.decode().split("#")[-2])
                out = exe_stdout.decode().strip()
            except BaseException as e:
                Warn(False, special_mask, "运行中断：", test_path, "\n", traceback.format_exc(), special_mask=special_mask)
                code_return.append(ExeResult("", -1, -1))
            else:
                code_return.append(ExeResult(out, input_times, ret))
            finally:
                process.kill()
            Warn(ret == 0, special_mask, "运行出错：", test_path, " ret: ", str(ret), special_mask=special_mask)

        return code_return, compile_succ

    def execute_code_thread(self, dir_path: str, files: list, num: int) -> [dict, int]:
        # 有bug，废弃
        exe_res_list = dict()
        compile_succ_list = dict()

        def act(mask: int, file_list: list) -> None:
            exe_res, total_num, cur_num = dict(), len(file_list), 1
            for name in file_list:
                Log(str(mask), "开始编译", name, "：[", str(cur_num), "/", str(total_num), "]")
                cur_num += 1
                p = PathFunc.join(dir_path, name)
                exe_res[name.split(".")[0]], cs = self.execute_code(p, str(mask))
            exe_res_list[mask] = exe_res
            compile_succ_list[mask] = compile_succ_list.get(mask, 0) + cs

        t, files_len, files_len_div = [], len(files), int((len(files) / num))
        for i in range(num):
            start = i * files_len_div
            end = files_len if i == num-1 else (i+1) * files_len_div
            t.append(threading.Thread(target=act, name="t"+str(i), args=(i, files[start:end])))
            t[i].start()
        ret_exe_res = dict()
        compile_succ = 0
        for i in range(num):
            t[i].join()
            ret_exe_res.update(exe_res_list[i])
            compile_succ += compile_succ_list[i]
        return ret_exe_res, compile_succ


if __name__ == "__main__":
    execute_code = ExecuteCode(is_save_intermediate_file=True)
    # execute_code.execute_code(PathFunc.to_linux(r"D:\homework\data_mining\homework4\Code\train\train\0ae1\00bb45560917420a.txt"))  #
    # execute_code.execute_code(r"D:\homework\data_mining\homework4\Code\train\train\0ae1\06b176c80a8e4bcf.txt")  # 会死循环
    # execute_code.execute_code(r"D:\homework\data_mining\homework4\Code\train\train\043e\0005efff92534ede.txt")  # gets
    # execute_code.execute_code(PathFunc.to_linux(r"D:\homework\data_mining\homework4\Code\train\train\083c\0056bb7a035549c8.txt"))  # get
    # execute_code.execute_code(PathFunc.to_linux(r"D:\homework\data_mining\homework4\Code\test\test\02aa0e6caa544c72.txt"))  # while(cin)
    # execute_code.execute_code(PathFunc.to_linux(r"D:\homework\data_mining\homework4\Code\test\test\f4d7dfe639d94661.txt"))  # 需要-1作为输入结束标志
    # execute_code.execute_code(PathFunc.to_linux(r"D:\homework\data_mining\homework4\Code\test\test\f39d279f7e9e479c.txt"))  # cin用到了!=
    execute_code.execute_code(PathFunc.to_linux(r"D:\homework\data_mining\homework4\Code\test\test\c0f1ea203fef4bc3.txt"))
