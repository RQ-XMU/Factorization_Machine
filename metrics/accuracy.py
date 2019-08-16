import numpy as np

class MRR:

    def __init__(self, length=20):
        self.length = length

    def reset(self):
        self.test = 0
        self.pos = 0

    def add(self, result, next_item):#这里的计算方式理解得不是很透彻，需要再理解下
        res = np.array(result[:self.length])
        self.test += 1
        if next_item in res:
            rank = np.where(res == next_item)[0].tolist()[0] + 1
            self.pos += ( 1.0/rank )

    def result(self):
        return ("MRR@" + str(self.length) + ": "), (self.pos/self.test)


class HitRate:
    def __init__(self, length=20):
        self.length = length

    def reset(self):
        self.test = 0
        self.hit = 0

    def add(self, result, next_item):
        self.test += 1
        if next_item in result[:self.length]:
            self.hit += 1

    def result(self):
        return ("HitRate@" + str(self.length) + ": "), (self.hit/self.test)




