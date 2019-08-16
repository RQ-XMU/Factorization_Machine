import time

def evaluate_sessions(pr, metrics, test_data):

    actions = len([i for i in test_data])
    sessions = len([i for i in test_data])
    count = 0
    print('START evaluation of ', actions, ' actions in ', sessions, ' sessions')

    sc = time.clock()
    st = time.time()

    time_sum = 0
    time_sum_clock = 0
    time_count = 0

    for m in metrics:
        m.reset()

    for user in test_data:

        if count % 100 == 0:
            print('   eval process: ', count, ' of ', actions, ' actions: ', ( count / actions * 100.0), ' % in', (time.time() - st), 's')

        val_item = test_data[user][0]
        test_item = test_data[user][1]
        crs = time.clock()
        trs = time.time()
        preds = pr.predict_next(val_item, user)
        time_sum_clock += time.clock() - crs
        time_sum += time.time() - trs
        time_count += 1

        for m in metrics:
            m.add(preds, test_item)
        count += 1
    print('END evaluation in ', (time.clock() - sc), 'c / ', (time.time() - st), 's')
    print('    avg rt ', (time_sum / time_count), 's / ', (time_sum_clock / time_count), 'c')
    res = []
    for m in metrics:
        res.append(m.result())

    return res






