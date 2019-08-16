import pandas as pd
import numpy as np
import random

class Dataset():
    def __init__(self):
        #2019-8-14
        #这里使用的u.data，其user编号从1开始，item编号从1开始，user：943，item:1682，且编号是连续的
        #因为之前见过一个数据集，其编号不是连续的，导致代码出现一些bug。
        df = pd.read_csv(r"DataSet/u.data", sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'time'], index_col=False)
        #这里对time进行归一化，是因为算法会考虑time因素。
        print("Normalizing temporal values...")
        mean = df['time'].mean()
        std = df['time'].std()
        self.ONE_YEAR = (60 * 60 * 24 * 365) / mean
        self.ONE_DAY = (60 * 60 * 24) / mean
        df['time'] = (df['time'] - mean) / std
        df.sort_values(['user_id', 'time'], inplace=True)
        df = df.reset_index(drop=True)
        num_users = df['user_id'].nunique()
        num_items = df['item_id'].nunique()
        self.num_users = num_users
        self.num_items = num_items

        print('Constructing datasets...')
        training_set = {}#2019-8-14：先当成implicit feedback来做
        train_set = {}
        val_set = {}
        test_set = {}
        item_feature = {}
        item_set_per_user = {}#产生推荐列表时排除用户已经交互过的item
        num_train_events = 0

        for row in df.itertuples(index=False):#2019-8-15:这里忘记在括号里加index=False，难怪会报错。
            if row[0] not in training_set:
                training_set[row[0]] = []
            training_set[row[0]].append((row[1], row[3]))
        for user in training_set:
            test_item, test_time = training_set[user].pop()
            val_item, val_time = training_set[user].pop()
            last_item, last_time = training_set[user][-1]
            val_set[user] = []
            val_set[user].append(last_item)
            val_set[user].append(val_item)
            test_set[user] = []
            test_set[user].append(val_item)
            test_set[user].append(test_item)
        for user in training_set:
            if user not in train_set:
                train_set[user] = []
                item_set_per_user[user] = []
            for i in range(1, len(training_set[user])):
                prev_item = training_set[user][i - 1][0]
                true_next_item = training_set[user][i][0]
                false_next_item = random.randrange(self.num_items) + 1
                while false_next_item == true_next_item:
                    false_next_item = random.randrange(self.num_items) + 1
                train_set[user].append((prev_item, true_next_item, false_next_item))
                item_set_per_user[user].append(prev_item)
                if i == len(training_set[user]) - 1:
                    item_set_per_user[user].append(true_next_item)

        for user in training_set:
            num_train_events += len(training_set[user])



        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.item_set_per_user = item_set_per_user
        self.num_train_events = num_train_events

        print('reading item demographics...')
        item_df = pd.read_csv(r"DataSet/u.item", encoding='ISO-8859-1',
                              sep='|', header=None, names=['movie_id', 'movie_title',
                                                           'release_date', 'unknown1', 'IMDb URL', 'unknown2', 'Action',
                                                           'Adventure',
                                                           'Animation', 'Children\'s',
                                                           'Comedy', 'Crime', 'Documentary',
                                                           'Drama', 'Fantasy', 'Film-Noir',
                                                           'Horror', 'Musical', 'Mystery',
                                                           'Romance', 'Sci-Fi', 'Thriller',
                                                           'War', 'Western'], index_col=False)
        item_df = item_df.set_index('movie_id')
        del item_df['unknown1']
        del item_df['IMDb URL']
        del item_df['release_date']
        del item_df['movie_title']
        self.item_df = item_df
        item_feats_array = self.item_df.values  # 这里得到的是二维数组
        for item_id in range(1, self.num_items+1):
            index = np.where(item_feats_array[item_id - 1, :] == 1)[0].tolist()  # 得到的是list
            item_feature[item_id] = [i + 4307 for i in index]

        self.item_feature_dim = item_feats_array.shape[1]
        self.item_feature = item_feature

