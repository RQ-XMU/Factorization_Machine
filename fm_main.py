import time
import numpy as np
import pandas as pd
import adapter as ad
import evaluation_test_last as eval_last
from metrics import accuracy as ac
'''
2019-8-16:
这种方法的计算复杂度好像很高。估计是自己代码实现的问题
'''
if __name__ == '__main__':
    export_csv = 'results/fm.csv'
    metric = []
    metric.append(ac.HitRate(20))
    metric.append(ac.HitRate(15))
    metric.append(ac.HitRate(10))
    metric.append(ac.HitRate(5))
    metric.append(ac.MRR(20))
    metric.append(ac.MRR(15))
    metric.append(ac.MRR(10))
    metric.append(ac.MRR(5))

    pr = ad.Adapter()

    start_time = time.time()
    print("  fit  ", "fm")
    pr.fit()

    res = {}

    res["fm"] = eval_last.evaluate_sessions(pr, metric, pr.instance.dataset.test_set)

    end_time = time.time()
    print("total time is: ", end_time - start_time, " s")

    for k, l in res.items():
        for e in l:
            print(k, ":", e[0], " ", e[1])

    if export_csv is not None:
        file = open(export_csv, 'w+')
        file.write('Metrics:')


        for k, l in res.items():
            for e in l:
                file.write(e[0])
                file.write(';')
            break
        file.write('\n')

        for k, l in res.items():
            file.write(k)
            file.write(";")
            for e in l:
                file.write(str(e[1]))
                file.write(';')
            file.write('\n')