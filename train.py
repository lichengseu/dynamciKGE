import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import json
import time
import datetime

import config

gpu_ids = [0, 1]

DEBUG = False


class DynamicKGE(nn.Module):
    def __init__(self, config):
        super(DynamicKGE, self).__init__()

        self.entity_gcn_weight = nn.Parameter(torch.FloatTensor(config.dim, config.dim))
        self.relation_gcn_weight = nn.Parameter(torch.FloatTensor(config.dim, config.dim))

        self.entity_emb = nn.Parameter(torch.FloatTensor(config.entity_total, config.dim))
        self.entity_context = nn.Parameter(torch.FloatTensor(config.entity_total + 1, config.dim))
        self.relation_emb = nn.Parameter(torch.FloatTensor(config.relation_total, config.dim))
        self.relation_context = nn.Parameter(torch.FloatTensor(config.relation_total + 1, config.dim))

        self.gate_entity = nn.Parameter(torch.FloatTensor(config.dim))
        self.gate_relation = nn.Parameter(torch.FloatTensor(config.dim))

        self.v_ent = nn.Parameter(torch.FloatTensor(config.dim))
        self.v_rel = nn.Parameter(torch.FloatTensor(config.dim))

        self.pht_o = dict()
        self.pr_o = dict()

        self.pht_sg = dict()
        self.pr_sg = dict()

        self._init_parameters()

    def _init_parameters(self):
        if config.init_with_transe:
            transe_entity_emb, transe_relation_emb = config.get_transe_embdding()
            self.entity_emb.data = transe_entity_emb
            self.relation_emb.data = transe_relation_emb
        else:
            nn.init.xavier_uniform_(self.entity_emb.data)
            nn.init.xavier_uniform_(self.relation_emb.data)

        entity_context_init = torch.FloatTensor(config.entity_total + 1, config.dim)
        nn.init.xavier_uniform_(entity_context_init)
        entity_context_init[-1] = torch.zeros(config.dim)
        self.entity_context.data = entity_context_init
        # original rate: entity:50, relation:5

        relation_context_init = torch.FloatTensor(config.relation_total + 1, config.dim)
        nn.init.xavier_uniform_(relation_context_init)
        relation_context_init[-1] = torch.zeros(config.dim)
        self.relation_context.data = relation_context_init

        nn.init.uniform_(self.gate_entity.data)
        nn.init.uniform_(self.gate_relation.data)
        nn.init.uniform_(self.v_ent.data)
        nn.init.uniform_(self.v_rel.data)

        stdv = 1. / math.sqrt(self.entity_gcn_weight.size(1))
        self.entity_gcn_weight.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.relation_gcn_weight.size(1))
        self.relation_gcn_weight.data.uniform_(-stdv, stdv)

    def _calc(self, h, t, r):
        return torch.norm(h + r - t, p=config.norm, dim=1)

    def get_adj_entity_vec(self, entity_vec_list, adj_entity_list):
        '''
        根据邻接实体列表按次序得到所有邻接实体的向量，并根据最大上下文数目进行补0
        :param adj_entity_list: 是将所有batch中所有list拼接而成的一个大list
        :return:
        '''
        adj_entity_vec_list = self.entity_context[adj_entity_list]
        adj_entity_vec_list = adj_entity_vec_list.view(-1, config.max_context_num, config.dim)

        return torch.cat((entity_vec_list.unsqueeze(1), adj_entity_vec_list), dim=1)

    def get_adj_relation_vec(self, relation_vec_list, adj_relation_list):
        adj_relation_vec_list = self.relation_context[adj_relation_list]
        adj_relation_vec_list = adj_relation_vec_list.view(-1, config.max_context_num, 2,
                                                           config.dim).cuda()  # 2是最长path的边数目
        # adj_relation_vec_list = torch.sum(adj_relation_vec_list, dim=2) / 2  # 将每个path求和平均
        adj_relation_vec_list = torch.sum(adj_relation_vec_list, dim=2)  # 将每个path求和

        return torch.cat((relation_vec_list.unsqueeze(1), adj_relation_vec_list), dim=1)

    def score(self, hs, o, target='entity'):
        os = torch.cat(tuple([o] * (config.max_context_num + 1)), dim=1).reshape(-1, config.max_context_num + 1, config.dim)
        tmp = F.relu(torch.mul(hs, os), inplace=False)  # batch x max x 2dim
        if target == 'entity':
            score = torch.matmul(tmp, self.v_ent)  # batch x max
        else:
            score = torch.matmul(tmp, self.v_rel)  # batch x max
        return score

    def gcn(self, A, H, target='entity'):
        '''
        :param A: subgraph adj matrix, size: batch_size x max_context_num x max_context_num
        :param H: subgraph vector, size: batch_size x max_context_num x dim
        :param target: target='entity' or 'relation'
        :return: gcn完之后的subgraph vector
        '''
        support = torch.matmul(A, H)
        if target == 'entity':
            output = F.relu(torch.matmul(support, self.entity_gcn_weight))
            # support = torch.matmul(A, output)
            # output = F.relu(torch.matmul(support, self.entity_gcn_weight))
        elif target == 'relation':
            output = F.relu(torch.matmul(support, self.relation_gcn_weight))
        return output  # 取gcn训练完之后的第一行向量，即目标向量

    def calc_subgraph_vec(self, o, adj_vec_list, target='entity'):
        '''

        :param adj_vec_list:
        :return:
        '''
        alpha = self.score(adj_vec_list, o, target)  # batch x max
        sg = torch.sum(torch.mul(torch.unsqueeze(alpha, dim=2), adj_vec_list), dim=1)  # batch x dim
        return sg

    def get_entity_context(self, entities):
        '''每个entity的context都对应着一个list，那么entities中的所有entity的context拼接成一个大list
        :param entities: [e, ..., e]
        :return:
        '''
        entities_context = []
        for e in entities:
            entities_context.extend(config.entity_adj_table.get(int(e), [config.entity_total] * config.max_context_num))
        return entities_context

    def get_relation_context(self, relations):
        '''
        :param relations:
        :return: relations_context: [ edges, ..., edges ], size: batch_size x max_context_num x 2
        edges === edge1, edge2 or edge1, pad or pad, pad
        '''
        relations_context = []
        for r in relations:
            relations_context.extend(
                config.relation_adj_table.get(int(r), [config.relation_total] * 2 * config.max_context_num))
        return relations_context

    def save_parameters(self, file_name, epoch):
        if not os.path.exists(config.res_dir):
            os.mkdir(config.res_dir)
        ent_f = open(config.res_dir + 'entity_' + file_name + str(epoch), "w")
        ent_f.write(json.dumps(self.pht_o))
        ent_f.close()

        rel_f = open(config.res_dir + 'relation_' + file_name + str(epoch), "w")
        rel_f.write(json.dumps(self.pr_o))
        rel_f.close()

        ent_sg_f = open(config.res_dir + 'entity_sg_' + file_name + str(epoch), 'w')
        ent_sg_f.write(json.dumps(self.pht_sg))
        ent_sg_f.close()

        rel_sg_f = open(config.res_dir + 'relation_sg_' + file_name + str(epoch), 'w')
        rel_sg_f.write(json.dumps(self.pr_sg))
        rel_sg_f.close()

        para2vec = {}
        para_dict = self.state_dict()
        for para_name in para_dict:
            para2vec[para_name] = para_dict[para_name].cpu().numpy().tolist()
        para_f = open(config.res_dir + 'all_parameters_' + file_name + str(epoch), 'w')
        para_f.write(json.dumps(para2vec))
        para_f.close()

    def save_phrt_o(self, pos_h, pos_r, pos_t, ph_o, pr_o, pt_o):
        for i in range(len(pos_h)):
            h = str(int(pos_h[i]))
            if h not in self.pht_o:
                self.pht_o[h] = ph_o[i].detach().cpu().numpy().tolist()
            # if h not in self.pht_sg:
            #     self.pht_sg[h] = ph_sg[i].detach().cpu().numpy().tolist()

        for i in range(len(pos_t)):
            t = str(int(pos_t[i]))
            if t not in self.pht_o:
                self.pht_o[t] = pt_o[i].detach().cpu().numpy().tolist()
            # if t not in self.pht_sg:
            #     self.pht_sg[t] = pt_sg[i].detach().cpu().numpy().tolist()

        for i in range(len(pos_r)):
            r = str(int(pos_r[i]))
            if r not in self.pr_o:
                self.pr_o[r] = pr_o[i].detach().cpu().numpy().tolist()
            # if r not in self.pr_sg:
            #     self.pr_sg[r] = pr_sg[i].detach().cpu().numpy().tolist()

    def forward(self, epoch, golden_triples, negative_triples, ph_A, pr_A, pt_A, nh_A, nr_A, nt_A):
        # multi golden and multi negative
        pos_h, pos_r, pos_t = golden_triples
        neg_h, neg_r, neg_t = negative_triples

        # target_vector_start = time.time()
        # target vector
        p_h = self.entity_emb[pos_h.type(torch.long)]
        p_t = self.entity_emb[pos_t.type(torch.long)]
        p_r = self.relation_emb[pos_r.type(torch.long)]
        n_h = self.entity_emb[neg_h.type(torch.long)]
        n_t = self.entity_emb[neg_t.type(torch.long)]
        n_r = self.relation_emb[neg_r.type(torch.long)]
        # print('target vector time: ' + str(target_vector_end - target_vector_start))

        # context_start = time.time()
        # context
        ph_adj_entity_list = self.get_entity_context(pos_h)
        pt_adj_entity_list = self.get_entity_context(pos_t)
        nh_adj_entity_list = self.get_entity_context(neg_h)
        nt_adj_entity_list = self.get_entity_context(neg_t)
        pr_adj_relation_list = self.get_relation_context(pos_r)
        nr_adj_relation_list = self.get_relation_context(neg_r)
        # context_end = time.time()
        # print('context time: ' + str(context_end - context_start))

        # context_vector_start = time.time()
        # context vectors
        ph_adj_entity_vec_list = self.get_adj_entity_vec(p_h, ph_adj_entity_list)
        pt_adj_entity_vec_list = self.get_adj_entity_vec(p_t, pt_adj_entity_list)
        nh_adj_entity_vec_list = self.get_adj_entity_vec(n_h, nh_adj_entity_list)
        nt_adj_entity_vec_list = self.get_adj_entity_vec(n_t, nt_adj_entity_list)
        pr_adj_relation_vec_list = self.get_adj_relation_vec(p_r, pr_adj_relation_list)
        nr_adj_relation_vec_list = self.get_adj_relation_vec(n_r, nr_adj_relation_list)
        # context_vector_end = time.time()
        # print('context vector time: ' + str(context_vector_end - context_vector_start))

        # gcn_vector_start = time.time()
        # GCN
        ph_adj_entity_vec_list = self.gcn(ph_A, ph_adj_entity_vec_list, target='entity')
        pt_adj_entity_vec_list = self.gcn(pt_A, pt_adj_entity_vec_list, target='entity')
        nh_adj_entity_vec_list = self.gcn(nh_A, nh_adj_entity_vec_list, target='entity')
        nt_adj_entity_vec_list = self.gcn(nt_A, nt_adj_entity_vec_list, target='entity')
        pr_adj_relation_vec_list = self.gcn(pr_A, pr_adj_relation_vec_list, target='relation')
        nr_adj_relation_vec_list = self.gcn(nr_A, nr_adj_relation_vec_list, target='relation')
        # gcn_vector_end = time.time()
        # print('gcn vector time: ' + str(gcn_vector_end - gcn_vector_start))

        # softmax_attention_start = time.time()
        # softmax and attention
        ph_sg = self.calc_subgraph_vec(p_h, ph_adj_entity_vec_list, target='entity')
        pt_sg = self.calc_subgraph_vec(p_t, pt_adj_entity_vec_list, target='entity')
        nh_sg = self.calc_subgraph_vec(n_h, nh_adj_entity_vec_list, target='entity')
        nt_sg = self.calc_subgraph_vec(n_t, nt_adj_entity_vec_list, target='entity')
        pr_sg = self.calc_subgraph_vec(p_r, pr_adj_relation_vec_list, target='relation')
        nr_sg = self.calc_subgraph_vec(n_r, nr_adj_relation_vec_list, target='relation')
        # softmax_attention_end = time.time()
        # print('softmax attention time: ' + str(softmax_attention_end - softmax_attention_start))

        # gate_start = time.time()
        # gate
        ph_o = torch.mul(F.sigmoid(self.gate_entity), p_h) + torch.mul(1 - F.sigmoid(self.gate_entity), ph_sg)
        pt_o = torch.mul(F.sigmoid(self.gate_entity), p_t) + torch.mul(1 - F.sigmoid(self.gate_entity), pt_sg)
        nh_o = torch.mul(F.sigmoid(self.gate_entity), n_h) + torch.mul(1 - F.sigmoid(self.gate_entity), nh_sg)
        nt_o = torch.mul(F.sigmoid(self.gate_entity), n_t) + torch.mul(1 - F.sigmoid(self.gate_entity), nt_sg)
        pr_o = torch.mul(F.sigmoid(self.gate_relation), p_r) + torch.mul(1 - F.sigmoid(self.gate_relation), pr_sg)
        nr_o = torch.mul(F.sigmoid(self.gate_relation), n_r) + torch.mul(1 - F.sigmoid(self.gate_relation), nr_sg)
        # gate_end = time.time()
        # print('gate time: ' + str(gate_end - gate_start))

        # concatenate
        # ph_o = torch.cat((p_h, ph_sg), dim=1)
        # pt_o = torch.cat((p_t, pt_sg), dim=1)
        # nh_o = torch.cat((n_h, nh_sg), dim=1)
        # nt_o = torch.cat((n_t, nt_sg), dim=1)
        # pr_o = torch.cat((p_r, pr_sg), dim=1)
        # nr_o = torch.cat((n_r, nr_sg), dim=1)

        # dot
        # ph_o = torch.mul(p_h, ph_sg)
        # pt_o = torch.mul(p_t, pt_sg)
        # nh_o = torch.mul(n_h, nh_sg)
        # nt_o = torch.mul(n_t, nt_sg)
        # pr_o = torch.mul(p_r, pr_sg)
        # nr_o = torch.mul(n_r, nr_sg)
        # pr_o = torch.mul(F.sigmoid(self.gate_relation), p_r) + torch.mul(1 - F.sigmoid(self.gate_relation), pr_sg)
        # nr_o = torch.mul(F.sigmoid(self.gate_relation), n_r) + torch.mul(1 - F.sigmoid(self.gate_relation), nr_sg)

        # plus
        # ph_o = p_h + ph_sg
        # pt_o =

        # score for loss
        p_score = self._calc(ph_o, pt_o, pr_o)
        n_score = self._calc(nh_o, nt_o, nr_o)

        if epoch % config.validation_step == 0:
            self.save_phrt_o(pos_h, pos_r, pos_t, ph_o, pr_o, pt_o)
            # self.save_phrt_o(pos_h, pos_r, pos_t, ph_o, pr_o, pt_o, ph_sg, pr_sg, pt_sg)
        else:
            self.pht_o.clear()
            self.pr_o.clear()
            self.pht_sg.clear()
            self.pr_sg.clear()

        return p_score, n_score


def main():
    print('preparing data...')
    phs, prs, pts, nhs, nrs, nts = config.prepare_data()
    print('preparing data complete')

    print('train starting...')
    dynamicKGE = DynamicKGE(config).cuda()

    # DataParallel
    # dynamicKGE.to(gpu_ids[0])
    # dynamicKGE = torch.nn.DataParallel(dynamicKGE, device_ids=gpu_ids)

    # distributed DataParallel
    # torch.distributed.init_process_group(backend="nccl", rank=0, world_size=2)
    # dynamicKGE = torch.nn.DataParallel(dynamicKGE, device_ids=gpu_ids)

    optimizer = optim.SGD(dynamicKGE.parameters(), lr=config.learning_rate)
    # optimizer = optim.Adam(dynamicKGE.parameters(), lr=config.learning_rate)
    criterion = nn.MarginRankingLoss(config.margin, True).cuda()

    for epoch in range(config.train_times):
        epoch_avg_loss = 0.0
        for batch in range(config.nbatchs):
            golden_triples, negative_triples = config.get_batch(batch, epoch, phs, prs, pts, nhs, nrs, nts)
            ph_A, pr_A, pt_A = config.get_batch_positive_A(golden_triples)
            nh_A, nr_A, nt_A = config.get_batch_negative_A(negative_triples)

            optimizer.zero_grad()

            p_scores, n_scores = dynamicKGE(epoch, golden_triples, negative_triples, ph_A, pr_A, pt_A, nh_A, nr_A, nt_A)
            y = torch.FloatTensor([-1]).cuda()
            loss = criterion(p_scores, n_scores, y)

            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

            if not DEBUG:
                epoch_avg_loss += (float(loss.item()) / config.nbatchs)

        if not DEBUG:
            print('### epoch average loss: ' + str(epoch_avg_loss))

        print('----------trained the ' + str(epoch) + ' epoch----------')
        if epoch % config.validation_step == 0:
            # dynamicKGE.module.save_parameters('parameters', epoch)
            dynamicKGE.save_parameters('parameters', epoch)

        if os.path.exists(config.res_dir + "early_stop.txt"):
            print('Early Stop.')
            break

    print('train ending...')


main()
