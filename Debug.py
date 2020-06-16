is_use_log = True
is_use_assert = True

warn_file_path = "build/warning_file"
is_warn_file_init = dict()


def connect_str(s: tuple) -> str:
    ret = ""
    for i in s:
        ret += i if i is str else str(i)
    return ret


def red(s: tuple) -> None:
    print("\033[31m", connect_str(s), "\033[0m")


def blue(s: tuple) -> None:
    print("\033[34m", connect_str(s), "\033[0m")


def yellow(s: tuple) -> None:
    print("\033[33m", connect_str(s), "\033[0m")


def Log(*info: str, is_force_use: bool = False) -> None:
    if is_force_use or is_use_log:
        blue(info)


def Warn(exp: bool, *info: str, is_make_invalid: bool = False, is_write_file: bool = True, special_mask: str = "") -> None:
    if not exp and is_use_assert and not is_make_invalid:
        yellow(info)
        if is_write_file:
            global is_warn_file_init
            if not is_warn_file_init.get(special_mask, False):
                is_warn_file_init[special_mask] = True
                with open(warn_file_path + special_mask + ".txt", "w") as f:
                    f.write(connect_str(info) + "\n")
            else:
                with open(warn_file_path + special_mask + ".txt", "a") as f:
                    f.write(connect_str(info) + "\n")


def Assert(exp: bool, *info: str, is_make_invalid: bool = False) -> None:
    if not exp and is_use_assert and not is_make_invalid:
        red(info)
        assert False

