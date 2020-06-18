import os
import re


def analysis_str(test1: str, test2: str, thresh: float = 0.5) -> bool:
    """
    有关于csv文件的一些假设，故这个函数只适用于本项目
    本函数主要通过看代码中的字符串是否有相似之处，来看是否是clone代码
    有点点效果，提升到了0.74
    """
    test_path_prefix = "./test/test/"
    with open(os.path.join(test_path_prefix, test1 + ".txt"), "r") as f:
        code1 = f.read()
    with open(os.path.join(test_path_prefix, test2 + ".txt"), "r") as f:
        code2 = f.read()

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
