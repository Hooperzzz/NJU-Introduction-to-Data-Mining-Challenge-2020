import pickle


class GetCompareResult:
    thresh = 0.8
    thresh1 = 0.01

    def __init__(self, test_index_map: str, network_result_path: str):
        with open(test_index_map, "rb") as f:
            self.test_index_map = pickle.load(f)
        with open(network_result_path, "rb") as f:
            self.network_result = pickle.load(f)

    def compare_method(self, list1, list2) -> bool:
        same, diff = 0, 0
        assert len(list1) == len(list2)
        for a, b in zip(list1, list2):
            if a >= self.thresh and b >= self.thresh:
                same += 1
            elif a < self.thresh <= b or b < self.thresh <= a:
                diff += 1
        if 1 <= same <= 1 and diff <= 0:
            res = True
        else:
            res = False
        return res

    def compare_method1(self, list1, list2) -> bool:
        # 不中用的比较方法
        dist = 0
        assert len(list1) == len(list2)
        for a, b in zip(list1, list2):
            dist += (a - b)**2
        dist /= len(list1)
        if dist < self.thresh1:
            res = True
        else:
            res = False
        return res

    def get(self, name1, name2) -> bool:
        return self.compare_method(self.network_result[self.test_index_map[name1]],
                                   self.network_result[self.test_index_map[name2]])
