import os
import re


def get_two_code(test1: str, test2: str) -> (str, str):
    test_path_prefix = "./test/test/"
    with open(os.path.join(test_path_prefix, test1 + ".txt"), "r") as f:
        code1 = f.read()
    with open(os.path.join(test_path_prefix, test2 + ".txt"), "r") as f:
        code2 = f.read()
    return code1, code2


def analysis_str(test1: str, test2: str, thresh: float = 0.5) -> bool:
    """
    有关于csv文件的一些假设，故这个函数只适用于本项目
    本函数主要通过看代码中的字符串是否有相似之处，来看是否是clone代码
    有点点效果，提升到了0.74
    """
    code1, code2 = get_two_code(test1, test2)
    # 首先需要把所有的字符串中的括号去除

    def get_str(code: str) -> set:
        """
        寻找代码中所有的字符串
        """
        pattern = [
            re.compile("\"(.*?)\""),
            re.compile("%[^a-zA-Z]*[a-zA-Z]"),
            re.compile("[ \n\r\t]+"),
            re.compile("%[0-9]*\.?[0-9]*lf")
        ]
        str_list1 = pattern[0].findall(pattern[1].sub(" ", pattern[3].sub(" ", code)))
        str_list2 = [s.replace(r"\n", " ") for s in str_list1]
        str_list3 = []
        for s in str_list2:
            s = pattern[2].sub(" ", s)
            if " " in s:
                str_list3.extend(s.split(" "))
            else:
                str_list3.append(s)
        ret = set(str_list3)
        if "" in ret:
            ret.remove("")
        return ret

    set1, set2 = get_str(code1), get_str(code2)
    len1, len2 = len(set1), len(set2)
    d1, d2 = len(set1.difference(set2)), len(set2.difference(set1))
    if len1 != 0 and len2 != 0 and \
            ((d1 == 0 and d2 == 0) or (d1 < len1 * thresh and d2 < len2 * thresh)):
        return True
    return False


def analysis_input_output_loop_num(test1: str, test2: str) -> bool:
    """
    通过输入和输出在的循环的深度，来判断是否相似
    """
    code1, code2 = get_two_code(test1, test2)
    code1 = re.sub("\"(.*?)\"", "", code1)
    code2 = re.sub("\"(.*?)\"", "", code2)
    code1 = re.sub("'(.*?)'", "", code1)
    code2 = re.sub("'(.*?)'", "", code2)

    def get_input_output_loop_num(code: str) -> (int, int, int, int):
        """
        仅是获得最大的，输入和输出所在循环深度数目
        判断scanf，cin，get
        判断printf，cout
        循环仅判断for和while
        """
        # 寻找所有的循环所笼罩的范围
        loop_loc = re.finditer(r"(while)|(for)", code)
        loop_scope = []
        for loc in loop_loc:
            # 首先找到左小括号
            cur = loc.span()[1]
            while True:
                if code[cur] == "(":
                    break
                cur += 1
            # 跳过小括号
            s_num = 1
            cur += 1
            term1 = cur
            while s_num:
                if code[cur] == "(":
                    s_num += 1
                    # print("!", code[cur], s_num)
                elif code[cur] == ")":
                    s_num -= 1
                    # print("@", code[cur], s_num)
                cur += 1
            del s_num
            # 开始找左大括号或者是分号（start从这里开始是为了把小括号中的输入放到外围中）
            start, end = cur, cur
            while True:
                if code[cur] == "{":
                    break
                elif code[cur] == ";":
                    end = cur
                    break
                cur += 1
            if start != end:
                loop_scope.append([start, end])
            # 确定大括号的范围
            b_num = 1
            cur += 1
            while b_num:
                if code[cur] == "{":
                    b_num += 1
                elif code[cur] == "}":
                    b_num -= 1
                cur += 1
            end = cur
            loop_scope.append([start, end])
        # 开始确定每个输入语句所处循环深度
        input_max_deep, input_deep_0 = 0, 0
        input_loc = re.finditer(r"(scanf)|(cin)|(get)|(getline)", code)
        for loc in input_loc:
            l_s, l_e = loc.span()
            deep = 0
            for start, end in loop_scope:
                if start < l_s < end:
                    assert start < l_e < end
                    deep += 1
            if deep == 0:
                input_deep_0 += 1
            input_max_deep = max(input_max_deep, deep)
        output_max_deep, output_deep_0 = 0, 0
        output_loc = re.finditer(r"(printf)|(cout)", code)
        for loc in output_loc:
            l_s, l_e = loc.span()
            deep = 0
            for start, end in loop_scope:
                if start < l_s < end:
                    assert start < l_e < end
                    deep += 1
            if deep == 0:
                output_deep_0 += 1
            output_max_deep = max(output_max_deep, deep)
        return input_max_deep, output_max_deep, input_deep_0, output_deep_0

    in1, out1, in01, out01 = get_input_output_loop_num(code1)
    in2, out2, in02, out02 = get_input_output_loop_num(code2)
    return in1 == in2 and out1 == out2 and in01 == in02 and out01 == out02
