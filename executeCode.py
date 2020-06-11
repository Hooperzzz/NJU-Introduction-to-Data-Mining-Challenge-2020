import subprocess


class ExecuteCode:
    work_path = ""

    code_file_name = ""
    exe_file_name = ""

    def __init__(self, command: str, is_save_intermediate_file: bool = False):
        self.command = command
        self.is_save_intermediate_file = is_save_intermediate_file

    def execute_code(self, test_path: str):
        # 首先把测试文件变成可执行的代码

        process = subprocess.Popen(self.command % (self.work_path + '.c', self.exe_file_name))
