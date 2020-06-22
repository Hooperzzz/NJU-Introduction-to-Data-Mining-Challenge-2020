from Debug import *


class GenerateAns:
    count = 0

    def __init__(self, ans_path: str, sample_path: str):
        self.ans_path = ans_path
        self.sample_path = sample_path
        with open(self.ans_path, "w") as f:
            f.write("id1_id2,predictions\n")
        self.f = open(self.sample_path, "r")
        self.f.readline()

        self.temp1, self.temp2 = "", ""

    def get(self) -> list:
        self.count += 1
        Log("当前进展", self.count)
        s = self.f.readline()
        if len(s) < 3:
            return None
        else:
            self.temp1, self.temp2 = s.split(",")[0].split("_")
            return [self.temp1, self.temp2]

    def add(self, ans: int):
        with open(self.ans_path, "a") as f:
            f.write(self.temp1 + "_" + self.temp2 + "," + str(ans) + "\n")


if __name__ == "__main__":
    g = GenerateAns("ans.csv", "sample_submission.csv")
    while True:
        t = g.get()
        if t is None:
            break
        print(t)
    pass
