import numpy as np
import pandas as pd
import fm as fm

class Adapter:

    def __init__(self, epochs=10):
        self.algo = fm
        self.instance = fm.FM()
        self.epochs = epochs

    def fit(self):
        self.instance.train()


    def predict_next(self, item_id, user_id):
        #2019-8-13
        #需要留意一个问题：
        #在进行推荐的时候，将用户已经交互过的item排除在外，否则算法很大概率会将用户已经交互过的item再次推荐
        #而这不是我们想要的效果。
        #算法需要达到的目的是将用户没有见过的item推荐给用户，这样才有意义。

        out = self.instance.top_k_recommendations(item_id, user_id)

        return out