import torch
import torch.nn as nn
import pickle
import os
import random
import math
import multiprocessing
import numpy as np
import json
import operator
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser(description='batch size, margin')
parser.add_argument('-b', '--batchsize', type=int, dest='batchsize', help='batch size', required=False, default=100)
parser.add_argument('-m', '--margin', type=float, dest='margin', help='margin', required=False, default=1.0)
parser.add_argument('-l', '--learning_rate', type=float, dest="learning_rate", help="learning rate", required=False, default=0.01)
parser.add_argument('-d', '--dimension', type=int, dest="dimension", help="dimension", required=False, default=100)
parser.add_argument('-n', '--norm', type=int, dest="norm", help="normalization", required=False, default=2)
parser.add_argument('-e', '--extra', type=str, dest="extra", help="extra information", required=False, default="")
args = parser.parse_args()


def read_file(train_file_name):
    train_data = []  # [(h, r, t)]
    with open(train_file_name) as f:
        lines = f.readlines()
        for line in lines:
            li = line.split()
            if len(li) == 3:
                train_data.append((int(li[0]), int(li[2]), int(li[1])))
    return train_data


def read_entity_total(entity_file_name):
    with open(entity_file_name) as f:
        entity_total = int(f.readline())
    return entity_total


def read_relation_total(relation_file_name):
    with open(relation_file_name) as f:
        relation_total = int(f.readline())
    return relation_total


dataset_v1 = 'YAGO3-10-part'
inverse_relation = False  # 是否考虑逆关系，(h,r,t)衍生出(t,r+relationTotal,h)

entity_total = read_entity_total(entity_file_name='./data/' + dataset_v1 + '/entity2id.txt')
if not inverse_relation:
    relation_total = read_relation_total(relation_file_name='./data/' + dataset_v1 + '/relation2id.txt')
    train_list = read_file(train_file_name='./data/' + dataset_v1 + '/train2id.txt')
    test_list = read_file(train_file_name='./data/' + dataset_v1 + '/test2id.txt')
    valid_list = read_file(train_file_name='./data/' + dataset_v1 + '/valid2id.txt')
else:
    relation_total = read_relation_total(relation_file_name='./data/' + dataset_v1 + '/inverse_relation2id.txt')
    train_list = read_file(train_file_name='./data/' + dataset_v1 + '/inverse_train2id.txt')
    test_list = read_file(train_file_name='./data/' + dataset_v1 + '/inverse_test2id.txt')
    valid_list = read_file(train_file_name='./data/' + dataset_v1 + '/inverse_valid2id.txt')


print('entity_total: ' + str(entity_total))
print('relation_total: ' + str(relation_total))
print('train_total: ' + str(len(train_list)))
print('test_total: ' + str(len(test_list)))
print('valid_total: ' + str(len(valid_list)))

max_context_num_constraint = True


# 经过一条边或两条边得到的所有路径 {entity: [[path]}
def get_1or2_path_from_head(head_ent, rel, entity_adj_table_with_rel):
    paths = {}  # actually second-order + first-order, {entity: [[edge]]}
    first_order_entity = set()
    first_order_relation = dict()

    if head_ent not in entity_adj_table_with_rel:
        return paths
    for tail_entity, relation in entity_adj_table_with_rel[head_ent]:
        first_order_entity.add(tail_entity)
        if relation != rel:     # 不包含关系自己
            if tail_entity in paths:
                paths[tail_entity].append([relation])
            else:
                paths[tail_entity] = [[relation]]

        if tail_entity in first_order_relation:
            first_order_relation[tail_entity].append(relation)
        else:
            first_order_relation[tail_entity] = [relation]

    for node in first_order_entity:
        if node not in entity_adj_table_with_rel:
            continue
        for tail_entity, relation in entity_adj_table_with_rel[node]:
            if tail_entity in paths:
                for r in first_order_relation[node]:
                    paths[tail_entity].append([r, relation])
            else:
                paths[tail_entity] = []
                for r in first_order_relation[node]:
                    paths[tail_entity].append([r, relation])

    return paths  # {entity: [[edge]]}


def find_relation_context(h, r, t, entity_adj_table_with_rel):
    tail_ent2paths = get_1or2_path_from_head(h, r, entity_adj_table_with_rel)  # 经过一条边或两条边得到的所有路径 {entity: [[edge]]}
    return tail_ent2paths.get(t, [])


def construct_adj_table(train_list):
    '''

    :param train_list:
    :return:
    entity_adj_table: {entity: [e, ..., e, pading, ...]}, key size: entity total, value size: max_context_num
    relation_adj_table: {relation: [e or pad, ..., e or pad]}, key size: relation total, value size: 2 * max_context_num
    '''
    entity_adj_table_with_rel = dict()  # {head_entity: [(tail_entity, relation)]}
    entity_adj_table = dict()  # {head_entity: [tail_entity]}
    relation_adj_table = dict()  # {relation: [[edge]]}

    for train_data in train_list:
        h, r, t = train_data

        if h not in entity_adj_table_with_rel:
            entity_adj_table_with_rel[h] = [(t, r)]
            entity_adj_table[h] = {t}
        else:
            entity_adj_table_with_rel[h].append((t, r))
            entity_adj_table[h].add(t)

    for train_data in train_list:
        h, r, t = train_data

        # 连接h t的一条边或两条边的路径上的所有边（关系）
        paths = find_relation_context(h, r, t, entity_adj_table_with_rel)

        if r not in relation_adj_table:
            relation_adj_table[r] = paths
        else:
            relation_adj_table[r] += paths

    for k, v in relation_adj_table.items():
        relation_adj_table[k] = set([tuple(i) for i in v])

    if max_context_num_constraint:
        max_context_num = 25
    else:
        max_context_num = 0
        for k, v in entity_adj_table.items():
            max_context_num = max(max_context_num, len(entity_adj_table[k]))
        for k, v in relation_adj_table.items():
            max_context_num = max(max_context_num, len(set([tuple(i) for i in relation_adj_table[k]])))

    entity_DAD = torch.FloatTensor(entity_total, max_context_num + 1, max_context_num + 1).cuda()
    relation_DAD = torch.FloatTensor(relation_total, max_context_num + 1, max_context_num + 1).cuda()

    for entity in range(entity_total):
        adj_num = max_context_num
        A = torch.eye(max_context_num + 1, max_context_num + 1).cuda()
        tmp = torch.ones(adj_num + 1).cuda()
        A[0, :adj_num + 1] = tmp
        A[:adj_num + 1, 0] = tmp
        D = np.eye(max_context_num + 1, max_context_num + 1)
        D[0][0] = max_context_num

        if entity in entity_adj_table:
            neighbours_list = list(entity_adj_table[entity])
            for index, neighbour in enumerate(neighbours_list):
                if neighbour not in entity_adj_table:
                    continue
                for index2, neighbour2 in enumerate(neighbours_list):
                    if index == index2:
                        continue
                    if neighbour2 in entity_adj_table[neighbour]:
                        A[index+1, index2+1] = 1
                        D[index+1][index+1] += 1

        D = np.linalg.inv(D)
        D = torch.Tensor(D).cuda()
        i = list(range(max_context_num + 1))
        D[i, i] = torch.sqrt(D[i, i])
        entity_DAD[entity] = D.mm(A).mm(D)

    for relation in range(relation_total):
        adj_num = max_context_num
        A = torch.eye(max_context_num + 1, max_context_num + 1).cuda()
        tmp = torch.ones(adj_num + 1).cuda()
        A[0, :adj_num + 1] = tmp
        A[:adj_num + 1, 0] = tmp

        D = np.eye(max_context_num + 1, max_context_num + 1)
        D[0][0] = max_context_num

        if relation in relation_adj_table:
            neighbours_set = relation_adj_table[relation]
            for index, neighbour in enumerate(neighbours_set):
                if len(neighbour) != 1:
                    continue
                if neighbour[0] not in relation_adj_table:
                    continue
                adj_set = relation_adj_table[neighbour[0]]
                for index2, neighbour2 in enumerate(neighbours_set):
                    if index == index2:
                        continue
                    if neighbour2 in adj_set:
                        A[index+1, index2+1] = 1
                        D[index+1][index+1] += 1
        D = np.linalg.inv(D)
        D = torch.Tensor(D).cuda()
        i = list(range(max_context_num + 1))
        D[i, i] = torch.sqrt(D[i, i])

        relation_DAD[relation] = D.mm(A).mm(D)

    for k, v in entity_adj_table.items():
        if max_context_num_constraint:
            res = list(v)
            if len(res) > max_context_num:
                res = res[:max_context_num]
        else:
            res = list(v)
        entity_adj_table[k] = res + [entity_total] * (max_context_num - len(res))  # 补padding
    for k, v in relation_adj_table.items():
        res = []
        for i in v:
            if len(res) == 2 * max_context_num:
                break
            if len(i) == 1:
                res.extend(list(i))
                res.append(relation_total)
            else:
                res.extend(list(i))

        relation_adj_table[k] = res + [relation_total] * 2 * (max_context_num - len(res) // 2)  # 补padding

    return entity_adj_table, relation_adj_table, max_context_num, entity_DAD, relation_DAD


def bern_sampling_prepare(train_list):
    head2count = dict()
    tail2count = dict()
    for h, r, t in train_list:
        head2count[h] = head2count.get(h, 0) + 1
        tail2count[t] = tail2count.get(t, 0) + 1

    hpt = 0.0  # head per tail
    for t, count in tail2count.items():
        hpt += count
    hpt /= len(tail2count)

    tph = 0.0
    for h, count in head2count.items():
        tph += count
    tph /= len(head2count)

    return tph, hpt


def one_negative_sampling(golden_triple, train_set, tph=0.0, hpt=0.0):
    negative_triple = tuple()
    h, r, t = golden_triple

    if not bern:  # uniform sampling
        while True:
            e = random.randint(0, entity_total - 1)
            is_head = random.randint(0, 1)
            if is_head:
                if (e, r, t) in train_set:
                    continue
                else:
                    negative_triple = (e, r, t)
                    break
            else:
                if (h, r, e) in train_set:
                    continue
                else:
                    negative_triple = (h, r, e)
                    break
    else:
        sampling_head_prob = tph / (tph + hpt)

        while True:
            e = random.randint(0, entity_total - 1)
            is_head = random.random() > sampling_head_prob
            if is_head:
                if (e, r, t) in train_set:
                    continue
                else:
                    negative_triple = (e, r, t)
                    break
            else:
                if (h, r, e) in train_set:
                    continue
                else:
                    negative_triple = (h, r, e)
                    break

    return negative_triple


def prepare_data():
    '''生成正例和负例'''
    phs = np.zeros(len(train_list), dtype=int)
    prs = np.zeros(len(train_list), dtype=int)
    pts = np.zeros(len(train_list), dtype=int)
    nhs = np.zeros((train_times, len(train_list)), dtype=int)
    nrs = np.zeros((train_times, len(train_list)), dtype=int)
    nts = np.zeros((train_times, len(train_list)), dtype=int)

    train_set = set(train_list)

    tph, hpt = bern_sampling_prepare(train_list)

    for i, golden_triple in enumerate(train_list):
        phs[i], prs[i], pts[i] = golden_triple

        for j in range(train_times):
            negative_triples = one_negative_sampling(golden_triple, train_set, tph, hpt)
            nhs[j][i], nrs[j][i], nts[j][i] = negative_triples

    return torch.IntTensor(phs).cuda(), torch.IntTensor(prs).cuda(), torch.IntTensor(pts).cuda(), torch.IntTensor(
        nhs).cuda(), torch.IntTensor(nrs).cuda(), torch.IntTensor(nts).cuda()


entity_adj_table, relation_adj_table, max_context_num, entity_A, relation_A = construct_adj_table(train_list)
train_times = 1001
validation_step = 40
norm = args.norm
learning_rate = args.learning_rate
batch_size = args.batchsize
nbatchs = math.ceil(len(train_list) / batch_size)  # 单例输入，等于训练数据的数目
dim = args.dimension
margin = args.margin
extra_info = args.extra
bern = True
init_with_transe = False
transe_model_file = 'TransE2.json'
res_dir = "./res2/%s_%s_%s_%s_%s_%s/" % (str(norm), str(batch_size), str(margin), str(dim), str(learning_rate), extra_info)

print('train_times: ' + str(train_times))
print('validation_step: ' + str(validation_step))
print('learning_rate: ' + str(learning_rate))
print('batch_size: ' + str(batch_size))
print('nbatchs: ' + str(nbatchs))
print('dim: ' + str(dim))
print('margin: ' + str(margin))
print('bern: ' + str(bern))
print('init_with_transe: ' + str(init_with_transe))


def get_batch(batch, epoch, phs, prs, pts, nhs, nrs, nts):
    r = min((batch + 1) * batch_size, len(train_list))

    return (phs[batch * batch_size: r], prs[batch * batch_size: r], pts[batch * batch_size: r]), \
           (nhs[epoch, batch * batch_size: r], nrs[epoch, batch * batch_size: r], nts[epoch, batch * batch_size: r])


def get_batch_positive_A(golden_triples):
    # multi golden and multi negative
    pos_h, pos_r, pos_t = golden_triples

    return entity_A[pos_h.cpu().numpy()], relation_A[pos_r.cpu().numpy()], entity_A[pos_t.cpu().numpy()]


def get_batch_negative_A(negative_triples):
    # multi golden and multi negative
    neg_h, neg_r, neg_t = negative_triples

    return entity_A[neg_h.cpu().numpy()], relation_A[neg_r.cpu().numpy()], entity_A[neg_t.cpu().numpy()]


def load_parameter(file_name, model='our', mode='input'):
    with open('./res/' + file_name, "r") as f:
        emb = json.loads(f.read())

    if model == 'transe':
        if mode == 'input':
            ent_embeddings = emb['ent_embeddings.weight']
            rel_embeddings = emb['rel_embeddings.weight']
            return torch.FloatTensor(ent_embeddings).cuda(), torch.FloatTensor(rel_embeddings).cuda()

        entity_emb = emb['entity_emb']
        relation_emb = emb['relation_emb']
        # entity_emb = emb['ent_embeddings.weight']
        # relation_emb = emb['rel_embeddings.weight']
        return entity_emb, relation_emb

    entity_emb = emb['entity_emb']
    relation_emb = emb['relation_emb']
    entity_context = emb['entity_context']
    relation_context = emb['relation_context']
    entity_gcn_weight = emb['entity_gcn_weight']
    relation_gcn_weight = emb['relation_gcn_weight']
    gate_entity = emb['gate_entity']
    gate_relation = emb['gate_relation']
    v_ent = emb['v_ent']
    v_rel = emb['v_rel']

    # give index, return embedding
    return torch.FloatTensor(entity_emb).cuda(), torch.FloatTensor(relation_emb).cuda(), torch.FloatTensor(
        entity_context).cuda(), torch.FloatTensor(relation_context).cuda(), torch.FloatTensor(
        entity_gcn_weight).cuda(), torch.FloatTensor(relation_gcn_weight).cuda(), torch.FloatTensor(
        gate_entity).cuda(), torch.FloatTensor(gate_relation).cuda(), torch.FloatTensor(v_ent).cuda(),torch.FloatTensor(
        v_rel).cuda()


def get_head_batch(golden_triple):
    head_batch = np.zeros((entity_total, 3), dtype=np.int32)
    head_batch[:, 0] = np.array(list(range(entity_total)))
    head_batch[:, 1] = np.array([golden_triple[1]] * entity_total)
    head_batch[:, 2] = np.array([golden_triple[2]] * entity_total)
    return head_batch


def get_tail_batch(golden_triple):
    tail_batch = torch.zeros(entity_total, 3, dtype=torch.int32)
    tail_batch[:, 0] = torch.tensor([golden_triple[0]] * entity_total)
    tail_batch[:, 1] = torch.tensor([golden_triple[1]] * entity_total)
    tail_batch[:, 2] = torch.tensor(list(range(entity_total)))
    return tail_batch


def get_transe_embdding():
    with open('./res/' + transe_model_file, "r") as f:
        emb = json.loads(f.read())

        entity_emb = emb['ent_embeddings.weight']
        relation_emb = emb['rel_embeddings.weight']
        return torch.FloatTensor(entity_emb).cuda(), torch.FloatTensor(relation_emb).cuda()
