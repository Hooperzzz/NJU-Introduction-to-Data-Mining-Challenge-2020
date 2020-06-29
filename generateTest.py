def line(s: int) -> str:
    return str(s) + "\r\n"


def end_signal() -> str:
    # 为测试添加一些可能的，结束标志
    # 因为测试数据大部分是oj，存在以0，-1等输入作为结束标志的情况
    # 因此，需要加一些可能的，表示输入结束的东西
    # 注意顺序，尽量把负数放在后面
    possible_end_signal = [0, '\n', -1]
    ret = ""
    for i in possible_end_signal:
        ret += line(i)
    return ret


def get_repetition_test(num: int, times: int) -> bytes:
    ret = ""
    for _ in range(times):
        ret += line(num)
    return (ret + end_signal()).encode()
