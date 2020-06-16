def join(*path: str, os: str = "linux") -> str:
    if os == "linux":
        ret = path[0]
        for p in path[1:]:
            ret += "/" + p
        return ret
    print("尚未实现", os, "下的路径拼合功能")
    assert False


def split(path: str, os: str = "linux") -> list:
    if os == "linux":
        return path.split("/")
    elif os == "windows":
        return path.split("\\")
    print("尚未实现", os, "下的路径拼合功能")
    assert False


def to_windows(path: str) -> str:
    return path.replace("/", "\\")


def to_linux(path: str) -> str:
    return path.replace("\\", "/")
