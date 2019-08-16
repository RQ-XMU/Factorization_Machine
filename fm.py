# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 11:21:25 2019

@author: Administrator
"""
from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
import math
import random
import re
import os
import glob
import sys
from time import time
import evaluation_val
import dataset


class FM(object):

    def __init__(self, reg=0.0025, learning_rate=0.05, annealing=1., init_sigma=1, k=32,  **kwargs):
        self.name = 'FM'
        self.dataset = dataset.Dataset()
        self.feature_dim = self.dataset.num_users + 2 * self.dataset.num_items + self.dataset.item_feature_dim

        self.reg = reg
        self.learning_rate = learning_rate  # self.learning_rate will change due to annealing.
        self.init_learning_rate = learning_rate  # self.init_learning_rate keeps the original value (for filename)
        self.annealing_rate = annealing
        self.init_sigma = init_sigma
        self.metrics = {'recall': {'direction': 1},
                        'precision': {'direction': 1},
                        'user_coverage': {'direction': 1},
                        'item_coverage': {'direction': 1},
                        'ndcg': {'direction': 1},
                        # 'blockbuster_share' : {'direction': -1}
                        }
        self.k = k #分解的维度

    def init_model(self):
        ''' Initialize the model parameters
        '''
        self.linear_factor = np.random.uniform(-0.1, 0.1, size=(self.feature_dim, 1)).astype(np.float32)
        self.interaction_factor = self.init_sigma * np.random.uniform(-0.1, 0.1, size=(self.feature_dim, self.k)).astype(np.float32)

    def _get_model_filename(self, epochs):
        '''Return the name of the file to save the current model
        '''
        filename = "fm_ne" + str(epochs) + "_lr" + str(self.init_learning_rate) + "_an" + str(
            self.annealing_rate) + "_k" + str(self.k) + "_reg" + str(
            self.reg) + "_ini" + str(self.init_sigma)
        return filename + ".npz"

    def get_pareto_front(self, metrics, metrics_names):
        costs = np.zeros((len(metrics[metrics_names[0]]), len(metrics_names)))
        for i, m in enumerate(metrics_names):
            costs[:, i] = np.array(metrics[m]) * self.metrics[m]['direction']
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(costs[is_efficient] >= c, axis=1)
        return np.where(is_efficient)[0].tolist()

    def _compute_validation_metrics(self, metrics):
        ev = evaluation_val.Evaluator()
        for userid in range(self.dataset.num_users):
            prev_item_id = self.dataset.val_set[userid + 1][0]
            top_k = self.top_k_recommendations(prev_item_id, userid +1)
            goal_item_id = self.dataset.val_set[userid + 1][1]
            ev.add_instance([goal_item_id], top_k)

        metrics['recall'].append(ev.average_recall())
        metrics['precision'].append(ev.average_precision())
        metrics['ndcg'].append(ev.average_ndcg())
        metrics['user_coverage'].append(ev.user_coverage())
        metrics['item_coverage'].append(ev.item_coverage())

        return metrics

    def train(self,
              max_time=np.inf,
              progress=100000,
              time_based_progress=False,
              autosave='Best',
              save_dir='mdl/',
              min_iter=100000,
              max_iter=2000000,
              max_progress_interval=np.inf,
              load_last_model=False,
              early_stopping=None,
              validation_metrics=['recall'],
              ):
        iterations = 0
        epochs_offset = 0
        if load_last_model:
            epochs_offset = self.load_last(save_dir)
        if epochs_offset == 0:
            self.init_model()
            # python 是动态语言，是在运行期执行，而不是在编译期执行，所以不管定义的方法是在父类还是子类，只要该对象有该方法就可以了，表面
        # 表面看起来就像父类调用子类的方法一样，链接：https://www.cnblogs.com/jianyungsun/p/6288047.html

        start_time = time()
        next_save = int(progress)
        train_costs = []  # 每min_iterations迭代后，对该min_iteration迭代内的所有error求个均值，在添加进来
        current_train_cost = []  # 记录一次sgd训练更新得到的误差，该List最大有min_iteration个sgd
        # 更新误差，达到min_iteration后，立即清空
        epochs = []
        metrics = {name: [] for name in self.metrics.keys()}
        filename = {}

        while (time() - start_time < max_time and iterations < max_iter):
            # 根据目前的理解，这里的收敛停止准则是认为给定一个最大迭代次数？或许是无法达到完美的收敛准则？
            # 应该就是人为设置一个最大迭代次数作为停止条件
            cost = self.training_step()  # 即一步SGD更新得到的ERROR

            current_train_cost.append(cost)

            # Check if it is time to save the model
            iterations += 1

            if time_based_progress:
                progress_indicator = int(time() - start_time)
            else:
                progress_indicator = iterations

            if progress_indicator >= next_save:

                if progress_indicator >= min_iter:

                    # Save current epoch
                    epochs.append(epochs_offset + iterations / self.dataset.num_train_events)

                    # Average train cost
                    train_costs.append(np.mean(current_train_cost))  # min_iterations的迭代次数的平均训练误差
                    current_train_cost = []  # min_iterations迭代次数后将这个list容器重置

                    # Compute validation cost 在min_iterations迭代次数后计算在验证集上的各种评价指标，
                    # 返回的为dict，关键字为各个评价指标，键值为相应的值
                    metrics = self._compute_validation_metrics(metrics)

                    # Print info
                    self._print_progress(iterations, epochs[-1], start_time, train_costs, metrics,
                                         validation_metrics)

                    # Save model
                    # 这一块真的不是很懂
                    # 这一块代码的作用理解：
                    # 因为要在验证集上计算大概epochs次的评估指标计算，每一次min_iterations完后
                    # 我们就会得到一个模型的参数，而每得到一次参数之后，程序需要将相应的参数保存到对应的文件里
                    # 由于这里是人为给定最大迭代次数作为收敛条件，所以越到后面的参数肯定是最好的
                    # 所以如果选择“best”参数，那么就需要在每一次得到新的参数文件之后，删除掉之前保存的旧的参数
                    # 文件，如果autosave = all，那么就简单很多了，直接保存每一次的参数文件就ok了。
                    #                    '''
                    #                    这里的代码片段可以这样理解：
                    #                    保留在验证集中评估效果最好的参数文件，如果最开始的参数是最好的，
                    #                    那么只会有一个保留文件的操作，如果最开始的参数在验证集上的表现不是最好的，
                    #                    那么会出现保留文件和删除文件的操作
                    #                    '''
                    run_nb = len(metrics[list(self.metrics.keys())[0]]) - 1  # 这里-1是为了和pareto_runs对应
                    # 因为这里下标也是从0开始的，这样下标的计数方式保持一致
                    # 因为要进行epochs次的数据集扫描，所以metrics中的某一个尺度对应的结果就有多个，
                    if autosave == 'All':
                        filename[run_nb] = save_dir + self._get_model_filename(round(epochs[-1], 3))
                        self.save(filename[run_nb])
                    elif autosave == 'Best':
                        pareto_runs = self.get_pareto_front(metrics, validation_metrics)
                        # 好像返回的是到目前的迭代轮次为止，一共在验证集上进行了多少次的验证计算，返回的是单元素的List
                        if run_nb in pareto_runs:
                            filename[run_nb] = save_dir + self._get_model_filename(round(epochs[-1], 3))
                            self.save(filename[run_nb])
                            to_delete = [r for r in filename if r not in pareto_runs]
                            for run in to_delete:
                                try:
                                    os.remove(filename[run])  # 这里的删除是将相应的文件删除掉
                                    print('Deleted ', filename[run])
                                except OSError:
                                    print('Warning : Previous model could not be deleted')
                                del filename[run]  # 这里的删除是将这个key_values对从filename字典中删除

                    if early_stopping is not None:
                        # Stop if early stopping is triggered for all the validation metrics
                        if all([early_stopping(epochs, metrics[m]) for m in validation_metrics]):
                            break

                        # Compute next checkpoint
                if isinstance(progress, int):
                    next_save += min(progress, max_progress_interval)
                else:
                    next_save += min(max_progress_interval, next_save * (progress - 1))

        best_run = np.argmax(
            np.array(metrics[validation_metrics[0]]) * self.metrics[validation_metrics[0]]['direction'])
        # 这里不是很懂？或者自己代入变量的时候有问题？
        return ({m: metrics[m][best_run] for m in self.metrics.keys()}, time() - start_time, filename[best_run])

    # 返回的是一个tuple，tuple的第一个元素记录了在验证上计算得到的最好结果，是一个dict，记录了每种尺度的最好结果

    def _print_progress(self, iterations, epochs, start_time, train_costs, metrics, validation_metrics):
        '''Print learning progress in terminal
        #这个函数的含义基本搞懂
        '''
        print(self.name, iterations, "batchs, ", epochs, " epochs in", time() - start_time, "s")
        print("Last train cost : ", train_costs[-1])
        for m in self.metrics:
            print(m, ': ', metrics[m][-1])
            if m in validation_metrics:
                print('Best ', m, ': ',
                      max(np.array(metrics[m]) * self.metrics[m]['direction']) * self.metrics[m]['direction'])
        print('-----------------')

    # 仿佛记得这个函数没用？
    def load_last(self, save_dir):
        '''Load last model from dir
        '''

        def extract_number_of_epochs(filename):
            m = re.search('_ne([0-9]+(\.[0-9]+)?)_', filename)
            return float(m.group(1))

        # Get all the models for this RNN
        file = save_dir + self._get_model_filename("*")
        file = np.array(glob.glob(file))

        if len(file) == 0:
            print('No previous model, starting from scratch')
            return 0

        # Find last model and load it
        last_batch = np.amax(np.array(map(extract_number_of_epochs, file)))
        last_model = save_dir + self._get_model_filename(last_batch)
        print('Starting from model ' + last_model)
        self.load(last_model)

        return last_batch

    def training_step(self):
        return self.sgd_step()

    def sgd_step(self):
        ''' Make one SGD update, given that the transition from prev_item to true_next exist in user history,
        But the transition prev_item to false_next does not exist.
        user, prev_item, true_next and false_next are all user or item ids.

        return error
        这里考虑的basket的size为1
        '''

        user = random.randrange(self.dataset.num_users)
        instance = self.dataset.train_set[user+1][random.randrange(len(self.dataset.train_set[user+1]))]
        prev_item = instance[0]
        positive_item = instance[1]
        negative_item = instance[2]

        positive_index = [user, prev_item + 942, positive_item + 943 + 1682 - 1]
        positive_index.extend(self.dataset.item_feature[positive_item])
        negative_index = [user, prev_item + 942, negative_item + 943 + 1682 - 1]
        negative_index.extend(self.dataset.item_feature[negative_item])

        linear_score_positive = np.sum(self.linear_factor[positive_index])
        linear_score_negative = np.sum(self.linear_factor[negative_index])

        interaction_score_positive = np.sum(np.square(np.sum(self.interaction_factor[positive_index], axis=0))) - \
            np.sum(np.square(self.interaction_factor[positive_index]))
        interaction_score_negative = np.sum(np.square(np.sum(self.interaction_factor[negative_index], axis=0))) - \
                                     np.sum(np.square(self.interaction_factor[negative_index]))

        preds_positive = linear_score_positive + 0.5 * interaction_score_positive
        preds_negative = linear_score_negative + 0.5 * interaction_score_negative
        delta = 1 - 1 / (1 + math.exp(min(10, max(-10, preds_negative - preds_positive))))

        temp_positive = np.zeros((self.feature_dim, 1)).astype(np.float32)
        for i in positive_index:
            temp_positive[i, 0] = 1.0
        temp_negative = np.zeros((self.feature_dim, 1)).astype(np.float32)
        for i in negative_index:
            temp_negative[i, 0] = 1.0
        #2019-8-16：这里的更新规则写错了，应该是参数的二范数，之前的是一范数的形式。
        #2019-8-16:不对，在FPMC的梯度表达式中，在进行参数更新的时候，用的不是参数的二范数，
        #不管是代码还是理论推导中，用的都不是二范数的形式，用的是参数本身的形式。
        #但是TransFM中用tf实现的时候，使用的是l2范数。
        #而且如果这里使用l2范数，会出错。不是代码逻辑的问题，是计算机计算的问题，会出现overflow的问题。
        self.linear_factor += self.learning_rate * (delta * (temp_positive - temp_negative) - self.reg * self.linear_factor)

        for i in range(self.k):
            interaction_factor_temp = self.interaction_factor[:, i]
            interaction_factor_temp = interaction_factor_temp.reshape((self.feature_dim, 1))

            interaction_factor_temp += self.learning_rate * (delta * (temp_positive * np.sum(interaction_factor_temp[positive_index]) \
                                                                            - temp_negative * np.sum(interaction_factor_temp[negative_index]) \
                                                                            - np.multiply(interaction_factor_temp, temp_positive) \
                                                                            + np.multiply(interaction_factor_temp, temp_negative))\
                                                                            - self.reg * interaction_factor_temp)
            self.interaction_factor[:, i] = interaction_factor_temp.flatten()

        return delta


    def top_k_recommendations(self, prev_item_id, user_id, k=30):
        #2019-8-13：
        #什么时候会用到这个函数？
        #只有在验证集上进行验证或者测试集上进行测试的时候才会调用这个函数。
        #在模型进行训练的时候是不会用到这个函数的。
        #记住将用户已经交互过的items从推荐列表中排除
        feature_vector_index = []
        item_score = []
        for item in range(1, self.dataset.num_items + 1):
            index_list = [user_id - 1, prev_item_id - 1 + 943, item - 1 + 943 + 1682]
            index_list.extend(self.dataset.item_feature[item])
            feature_vector_index.append(index_list)
            score = np.sum(self.linear_factor[feature_vector_index[item-1]]) + 0.5 * \
                    (np.sum(np.square(np.sum(self.interaction_factor[feature_vector_index[item-1]], axis=0))) - \
                     np.sum(np.square(self.interaction_factor[feature_vector_index[item-1]])))
            item_score.append(score)
        item_viewed = [item - 1 for item in set(self.dataset.item_set_per_user[user_id])]

        #这句代码的目的是将用户已经交互过的item从推荐列表中排除
        for item in item_viewed:
            item_score[item] = -np.inf
        #2019-8-13
        #这里还有个问题，对于验证集而言，排除掉训练集中的item就行了，但对于测试集而言，
        #按照目前的代码逻辑，岂不是需要把验证集中的和测试集中的Item都排除掉，虽然验证集只有one item。
        #这个问题后面再解决

        #find top k according to item_score
        # （这里的Item_id是item_to_index后的item_id（得找个机会把这个逻辑改通）
        #得注意下标的问题，到底哪里是从0开始的，哪里是从1开始的，主要还是数据集处理那边导致这个问题
        return [i + 1 for i in list(np.argpartition(-np.array(item_score), range(k))[:k])]#返回的是下标，刚好对应item_id
        # output = np.dot(uv, self.V_item_user.T) + np.dot(self.V_prev_next[last_item, :], self.V_next_prev.T)
        #
        # # Put low similarity to viewed items to exclude them from recommendations
        # output[[i[0] for i in sequence]] = -np.inf
        # output[exclude] = -np.inf
        #
        # # find top k according to output
        # return list(np.argpartition(-output, range(k))[:k])

    def save(self, filename):
        '''Save the parameters of a network into a file
        '''
        print('Save model in ' + filename)
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        np.savez(filename, linear_factor=self.linear_factor, interaction_factor=self.interaction_factor)

    def load(self, filename):
        '''Load parameters values form a file
        '''
        f = np.load(filename)
        self.linear_factor = f['linear_factor']
        self.interaction_factor = f['interaction_factor']
