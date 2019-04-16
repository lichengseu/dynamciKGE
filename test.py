import numpy as np
import json
from multiprocessing import Pool

import config


def _calc(h, t, r):
    return float(np.linalg.norm(np.array(h) + np.array(r) - np.array(t), ord=1))


def sub_predict(batch):
    pos_hs = batch[:, 0]
    pos_rs = batch[:, 1]
    pos_ts = batch[:, 2]

    p_score = []

    for i in range(len(pos_hs)):
        pos_h = str(int(pos_hs[i]))
        pos_r = str(int(pos_rs[i]))
        pos_t = str(int(pos_ts[i]))

        # score for loss
        p_score.append(_calc(entity2vec[pos_h], entity2vec[pos_t], relation2vec[pos_r]))

    return p_score


def predict(batch):
    p_score = []

    ####  multi process
    if multiprocess_switch:
        small_batch_size = 5000
        size = len(batch[:, 0])
        i = 0
        p = Pool()
        results = []
        while i < size:
            result = p.apply_async(sub_predict, args=(batch[i: min(size, i + small_batch_size)],))
            results.append(result)
            i += small_batch_size
        p.close()
        p.join()
        for result in results:
            p_score.extend(result.get())
        return p_score

    ####  single process
    pos_hs = batch[:, 0]
    pos_rs = batch[:, 1]
    pos_ts = batch[:, 2]
    for i in range(len(pos_hs)):
        pos_h = str(int(pos_hs[i]))
        pos_r = str(int(pos_rs[i]))
        pos_t = str(int(pos_ts[i]))

        # score for loss
        p_score.append(_calc(entity2vec[pos_h], entity2vec[pos_t], relation2vec[pos_r]))

    return p_score


def test_head(golden_triple):
    head_batch = config.get_head_batch(golden_triple)
    value = list(predict(head_batch))
    li = np.argsort(value)
    res = 0
    sub = 0
    for pos, val in enumerate(li):
        if val == golden_triple[0]:
            res = pos + 1
            break
        if (val, golden_triple[1], golden_triple[2]) in train_set:
            sub += 1

    del head_batch
    del value
    del li

    return res, res - sub


def test_tail(golden_triple):
    tail_batch = config.get_tail_batch(golden_triple)
    value = list(predict(tail_batch))
    li = np.argsort(value)
    res = 0
    sub = 0
    for pos, val in enumerate(li):
        if val == golden_triple[2]:
            res = pos + 1
            break
        if (golden_triple[0], golden_triple[1], val) in train_set:
            sub += 1

    del tail_batch
    del value
    del li

    return res, res - sub


def test_link_prediction(test_list):
    '''
    遍历所有三元组，对于每个三元组，替换头实体为所有其他实体，再判断正确三元组所在的位置，记录下来
    替换尾实体为所有其他实体，重复同样操作
    '''
    # test_list = read_test_file()
    # test_list = read_file(train_file_name='./data/YAGO3-10-part/test2id.txt')
    test_total = len(test_list)

    l_mr = 0
    r_mr = 0

    l_mrr = 0.0
    l_mrr_filter = 0.0
    r_mrr = 0.0
    r_mrr_filter = 0.0

    l_hit1 = 0
    l_hit3 = 0
    l_hit10 = 0
    r_hit1 = 0
    r_hit3 = 0
    r_hit10 = 0

    l_mr_filter = 0
    r_mr_filter = 0
    l_hit1_filter = 0
    l_hit3_filter = 0
    l_hit10_filter = 0
    r_hit1_filter = 0
    r_hit3_filter = 0
    r_hit10_filter = 0


    for i, golden_triple in enumerate(test_list):
        print('test ---' + str(i) + '--- triple')
        l_pos, l_filter_pos = test_head(golden_triple)
        r_pos, r_filter_pos = test_tail(golden_triple)  # position, 1-based

        print(golden_triple, end=': ')
        print('l_pos=' + str(l_pos), end=', ')
        print('l_filter_pos=' + str(l_filter_pos), end=', ')
        print('r_pos=' + str(r_pos), end=', ')
        print('r_filter_pos=' + str(r_filter_pos), end='\n')

        l_mr += l_pos
        r_mr += r_pos

        l_mrr += 1.0 / l_pos
        l_mrr_filter += 1.0 / l_filter_pos

        r_mrr += 1.0 / r_pos
        r_mrr_filter += 1.0 / r_filter_pos

        if l_pos <= 1:
            l_hit1 += 1
        if l_pos <= 3:
            l_hit3 += 1
        if l_pos <= 10:
            l_hit10 += 1

        if r_pos <= 1:
            r_hit1 += 1
        if r_pos <= 3:
            r_hit3 += 1
        if r_pos <= 10:
            r_hit10 += 1

        ####################
        l_mr_filter += l_filter_pos
        r_mr_filter += r_filter_pos

        if l_filter_pos <= 1:
            l_hit1_filter += 1
        if l_filter_pos <= 3:
            l_hit3_filter += 1
        if l_filter_pos <= 10:
            l_hit10_filter += 1

        if r_filter_pos <= 1:
            r_hit1_filter += 1
        if r_filter_pos <= 3:
            r_hit3_filter += 1
        if r_filter_pos <= 10:
            r_hit10_filter += 1

    l_mr /= test_total
    r_mr /= test_total

    l_mrr /= test_total
    r_mrr /= test_total

    l_hit1 /= test_total
    l_hit3 /= test_total
    l_hit10 /= test_total
    r_hit1 /= test_total
    r_hit3 /= test_total
    r_hit10 /= test_total

    l_mr_filter /= test_total
    r_mr_filter /= test_total

    l_mrr_filter /= test_total
    r_mrr_filter /= test_total

    l_hit1_filter /= test_total
    l_hit3_filter /= test_total
    l_hit10_filter /= test_total
    r_hit1_filter /= test_total
    r_hit3_filter /= test_total
    r_hit10_filter /= test_total

    print('metric\tMRR\tMR\thit@10\thit@3\thit@1')
    print('head(raw)\t' + str(l_mrr) + '\t' + str(l_mr) + '\t' + str(l_hit10) + '\t' + str(
        l_hit3) + '\t' + str(l_hit1))
    print('tail(raw)\t' + str(r_mrr) + '\t' + str(r_mr) + '\t' + str(r_hit10) + '\t' + str(
        r_hit3) + '\t' + str(r_hit1))
    print('Average(raw)\t' + str((l_mrr + r_mrr) / 2) + '\t' + str((l_mr + r_mr) / 2) + '\t' + str(
        (l_hit10 + r_hit10) / 2) + '\t' + str(
        (l_hit3 + r_hit3) / 2) + '\t' + str((l_hit1 + r_hit1) / 2))

    print('head(filter)\t' + str(l_mrr_filter) + '\t' + str(l_mr_filter) + '\t' + str(
        l_hit10_filter) + '\t' + str(
        l_hit3_filter) + '\t' + str(l_hit1_filter))
    print('tail(filter)\t' + str(r_mrr_filter) + '\t' + str(r_mr_filter) + '\t' + str(
        r_hit10_filter) + '\t' + str(
        r_hit3_filter) + '\t' + str(r_hit1_filter))
    print('Average(filter)\t' + str((l_mrr_filter + r_mrr_filter) / 2) + '\t' + str(
        (l_mr_filter + r_mr_filter) / 2) + '\t' + str(
        (l_hit10_filter + r_hit10_filter) / 2) + '\t' + str(
        (l_hit3_filter + r_hit3_filter) / 2) + '\t' + str((l_hit1_filter + r_hit1_filter) / 2))

multiprocess_switch = False

entity2vec = json.loads(open('res2/1_1000_10.0_100_0.1_test_our_transe/entity_parameters1000').read())
relation2vec = json.loads(open('res2/1_1000_10.0_100_0.1_test_our_transe/relation_parameters1000').read())
train_set = set(config.train_list)
print('test link prediction starting...')
test_link_prediction(config.test_list)
print('test link prediction ending...')
