def get_repetition_test(num: int, times: int) -> bytes:
    ret = ""
    for _ in range(times):
        ret += str(num) + "\r\n"
    return ret.encode()
