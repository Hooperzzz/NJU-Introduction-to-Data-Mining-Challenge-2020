"""
为测试加上必要的，代码头部，并替换输入函数以统计输入次数
以input_data作为输入，获取输出，输入次数以及代码执行结果
"""
import subprocess
import generateTest


class ExecuteCode:
    FS = r"/"  # 路径分隔符
    work_path = "build"
    intermediate_file_dir = work_path + FS + "intermediate_file"

    code_file_name = "test.cc"
    exe_file_name = "test"
    code_file_front = \
        "#include<bits/stdc++.h>\r\n" \
        "#define GET_INPUT_NUM cerr << INPUT_NUM << \"#\"\r\n" \
        "using namespace std;\r\n" \
        "int INPUT_NUM = 0;\r\n" \
        "\r\n" \
        "int SCANF(const char *fmt , ...){\r\n" \
        "\tint ret;\r\n" \
        "\tva_list ap;\r\n" \
        "\tva_start(ap , fmt);\r\n" \
        "\tINPUT_NUM++;\r\n" \
        "\tret = vscanf(fmt,ap);\r\n" \
        "\tva_end(ap);\r\n" \
        "\tGET_INPUT_NUM;\r\n" \
        "\treturn ret;\r\n" \
        "}\r\n" \
        "\r\n" \
        "#define CIN_OPERATOR(Type)" \
        "  friend CIN &operator>>(CIN &in, Type &obj){" \
        "    INPUT_NUM++;" \
        "    cin >> obj;" \
        "    GET_INPUT_NUM;" \
        "  }\r\n" \
        "class CIN {\r\n" \
        "\tCIN_OPERATOR(int);\r\n" \
        "\tCIN_OPERATOR(unsigned);\r\n" \
        "\tCIN_OPERATOR(short);\r\n" \
        "\tCIN_OPERATOR(unsigned short);\r\n" \
        "\tCIN_OPERATOR(long);\r\n" \
        "\tCIN_OPERATOR(unsigned long );\r\n" \
        "\tCIN_OPERATOR(long long);\r\n" \
        "\tCIN_OPERATOR(unsigned long long);\r\n" \
        "\tCIN_OPERATOR(char);\r\n" \
        "\tCIN_OPERATOR(unsigned char);\r\n" \
        "\tCIN_OPERATOR(float);\r\n" \
        "\tCIN_OPERATOR(double);\r\n" \
        "\tCIN_OPERATOR(long double);\r\n" \
        "\tCIN_OPERATOR(bool);\r\n" \
        "} CIN_OBJ;\r\n" \
        "\r\n" \
        "#define scanf SCANF\r\n" \
        "#define cin CIN_OBJ\r\n"

    input_data = [
        generateTest.get_repetition_test(1, 100),
        generateTest.get_repetition_test(3, 100)
    ]

    def __init__(self, is_save_intermediate_file: bool = False):
        self.is_save_intermediate_file = is_save_intermediate_file

    def execute_code(self, test_path: str) -> list:
        # 首先把测试文件变成可执行的代码
        new_code = ""
        with open(test_path, 'r') as f:
            new_code = self.code_file_front + f.read()
        new_code.replace("void main()", "int main()")  # 特殊处理：void main()需要变成int main()
        # 输出代码文件
        code_path = self.work_path + self.FS + self.code_file_name
        with open(code_path, 'w') as f:
            f.write(new_code)
        # 当需要留存中间生成文件时使用
        if self.is_save_intermediate_file:
            test_name = test_path.split('/')[-1]
            with open(self.intermediate_file_dir + '/' + test_name + '.cc', 'w') as f:
                f.write(new_code)
        # 编译出可执行文件
        exe_path = self.work_path + self.FS + self.exe_file_name
        compile_result = subprocess.run(["wsl", "g++", code_path, "-o", exe_path],
                                        stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE,
                                        shell=True)
        assert compile_result.returncode == 0
        # 运行代码，记录输入以及输出
        code_return = []
        for t in self.input_data:
            process = subprocess.Popen(["wsl", exe_path],
                                       stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       shell=True)
            exe_stdout, exe_stderr = process.communicate(input=t, timeout=1000)
            ret = process.poll()
            assert ret == 0  # 当代码执行不成功时，需要检查原因
            input_times = 0 if exe_stderr.decode() == "" else int(exe_stderr.decode().split("#")[-2])
            code_return.append([exe_stdout, input_times, ret])

        return code_return


if __name__ == "__main__":
    execute_code = ExecuteCode()
    execute_code.execute_code(r"D:\homework\data_mining\homework4\Code\test\test\00a7a1f7dcc0431b.txt")
