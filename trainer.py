## Author: Han Wu (han.wu@sydney.edu.au)

import os
import sys
import json
import torch
import shutil
import logging
import itertools
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from torch.autograd import Variable
from info_nce import InfoNCE
from models import Hire
formatter = logging.Formatter('%(asctime)s %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

best_hits10 = 0; best_hits5 = 0; best_hits1 = 0
best_hits10_v2 = 0; best_hits5_v2 = 0; best_hits1_v2 = 0

def setup_logger(name, log_file, level=logging.INFO):
    
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())

    return logger

class Trainer:
    def __init__(self, data_loaders, dataset, parameter):
        self.parameter = parameter
        self.dataset = dataset
        self.train_data_loader = data_loaders[0]
        self.dev_data_loader = data_loaders[1]
        self.test_data_loader = data_loaders[2]
        self.few = parameter['few']
        self.num_query = parameter['num_query']
        self.batch_size = parameter['batch_size']
        self.learning_rate = parameter['learning_rate']
        self.early_stopping_patience = parameter['early_stopping_patience']
        self.max_neighbor = parameter['max_neighbor']
        self.max_nn_meta = parameter['max_nn_meta']
        self.embed_model = parameter['embed_model']
        self.epoch = parameter['epoch']
        self.print_epoch = parameter['print_epoch']
        self.eval_epoch = parameter['eval_epoch']
        self.checkpoint_epoch = parameter['checkpoint_epoch']
        self.device = parameter['device']
        self.data_path = parameter['data_path']
        self.ent2id = dataset['ent2id']
        self.rel2id = dataset['rel2id']
        self.num_ents = len(self.ent2id.keys())
        self.load_embed()
        self.num_symbols = len(self.symbol2id.keys()) - 1  # one for 'PAD'
        self.pad_id = self.num_symbols
        degrees = self.build_connection(max_=self.max_neighbor)
        self.hire = Hire(dataset, parameter, self.num_symbols, embed=self.symbol2emb)
        self.hire.to(self.device)
        self.optimizer = torch.optim.Adam(self.hire.parameters(), self.learning_rate)
        self.state_dir = os.path.join(self.parameter['state_dir'], self.parameter['prefix'])
        if not os.path.isdir(self.state_dir):
            os.makedirs(self.state_dir)
        self.ckpt_dir = os.path.join(self.parameter['state_dir'], self.parameter['prefix'], 'checkpoint')
        if not os.path.isdir(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.state_dict_file = ''
        
        if not os.path.exists('./logs/'):
            os.makedirs('./logs')        
        if not os.path.exists('./loss-logs/'):
            os.makedirs('./loss-logs')
        self.logger = setup_logger('logger', './logs/{}.txt'.format(self.parameter['prefix']))
        self.loss_logger = setup_logger('loss_logger', './loss-logs/{}.txt'.format(self.parameter['prefix']))
        
        self.logger.info("---------Parameters---------")
        for k, v in self.parameter.items():
            self.logger.info(k + ': ' + str(v))
        self.logger.info("----------------------------")
        
        # load state_dict and params
        if parameter['step'] in ['test', 'dev']:
            self.reload()
    
    def load_embed(self):
        symbol_id = {}
        rel2id = json.load(open(self.data_path + '/relation2ids'))
        ent2id = json.load(open(self.data_path + '/ent2ids'))
        if self.embed_model in ['DistMult', 'TransE', 'ComplEx', 'RESCAL']:
            ent_embed = np.loadtxt(self.data_path + '/emb' + '/entity2vec.' + self.embed_model)
            rel_embed = np.loadtxt(self.data_path + '/emb' + '/relation2vec.' + self.embed_model)

            if self.embed_model == 'ComplEx':
                ent_mean = np.mean(ent_embed, axis=1, keepdims=True)
                ent_std = np.std(ent_embed, axis=1, keepdims=True)
                rel_mean = np.mean(rel_embed, axis=1, keepdims=True)
                rel_std = np.std(rel_embed, axis=1, keepdims=True)
                eps = 1e-3
                ent_embed = (ent_embed - ent_mean) / (ent_std + eps)
                rel_embed = (rel_embed - rel_mean) / (rel_std + eps)

            assert ent_embed.shape[0] == len(ent2id.keys())
            assert rel_embed.shape[0] == len(rel2id.keys())

            i = 0
            embeddings = []
            for key in rel2id.keys():
                if key not in ['','OOV']:
                    symbol_id[key] = i
                    i += 1
                    embeddings.append(list(rel_embed[rel2id[key],:]))

            for key in self.ent2id.keys():
                if key not in ['', 'OOV']:
                    symbol_id[key] = i
                    i += 1
                    embeddings.append(list(ent_embed[self.ent2id[key],:]))

            symbol_id['PAD'] = i
            embeddings.append(list(np.zeros((rel_embed.shape[1],))))
            embeddings = np.array(embeddings)
            assert embeddings.shape[0] == len(symbol_id.keys())

            self.symbol2id = symbol_id
            self.id2symbol = {value : key for (key, value) in symbol_id.items()}
            self.symbol2emb = embeddings
    
    def build_connection(self, max_=100):
        self.connections = (np.ones((self.num_ents, max_, 2)) * self.pad_id).astype(int)
        self.e1_rele2 = defaultdict(list)
        self.e1_degrees = defaultdict(int)
        with open(self.data_path + '/path_graph') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                e1,rel,e2 = line.rstrip().split()
                self.e1_rele2[e1].append((self.symbol2id[rel], self.symbol2id[e2]))
                self.e1_rele2[e2].append((self.symbol2id[rel+'_inv'], self.symbol2id[e1]))
        
        degrees = {}
        for ent, id_ in self.ent2id.items():
            neighbors = self.e1_rele2[ent]
            if len(neighbors) > max_:
                neighbors = neighbors[:max_]
            degrees[ent] = len(neighbors)
            self.e1_degrees[id_] = len(neighbors) # add one for self conn
            for idx, _ in enumerate(neighbors):
                self.connections[id_, idx, 0] = _[0]
                self.connections[id_, idx, 1] = _[1]

        return degrees
    
    def get_task_meta(self, task, iseval):
        task_meta = []
        for triplet_list in task:
            triplet_left = [self.ent2id[few[0]] for batch in triplet_list for few in batch]
            triplet_right = [self.ent2id[few[2]] for batch in triplet_list for few in batch]
            if not iseval:
                triplet_meta_left = [triplet_left[i:i+self.few] for i in range(0, len(triplet_left), self.few)]
                triplet_meta_left = np.transpose(np.array(triplet_meta_left)).tolist()
                triplet_meta_right = [triplet_right[i:i+self.few] for i in range(0, len(triplet_right), self.few)]
                triplet_meta_right = np.transpose(np.array(triplet_meta_right)).tolist()
            else:
                triplet_meta_left = [[0] for i in range(self.few)]
                triplet_meta_right = [[0] for i in range(self.few)]
            task_meta.append([self.get_meta(triplet_meta_left[i], triplet_meta_right[i]) for i in range(len(triplet_meta_left))] )
            
        return task_meta
    
    def get_meta(self, left, right):
        left_connections = Variable(torch.LongTensor(np.stack([self.connections[_, :, :] for _ in left], axis=0))).to(self.device)
        left_degrees = Variable(torch.LongTensor([self.e1_degrees[_] for _ in left])).to(self.device)
        right_connections = Variable(torch.LongTensor(np.stack([self.connections[_, :, :] for _ in right], axis=0))).to(self.device)
        right_degrees = Variable(torch.LongTensor([self.e1_degrees[_] for _ in right])).to(self.device)
    
        return (left_connections, left_degrees, right_connections, right_degrees)
    
    def get_negative_meta(self, meta):
        negative_meta = []
        for few_id in range(self.few):
            left_connections, left_degrees, right_connections, right_degrees = meta[few_id]
            false_left_connections, false_right_connections = left_connections.clone(), right_connections.clone()
            triplet_num, neighbor_num = false_left_connections.shape[0], false_left_connections.shape[1]
            false_left_connections[:, :, 0] = torch.randint(0, len(self.rel2id), (triplet_num, neighbor_num))
            false_right_connections[:, :, 0] = torch.randint(0, len(self.rel2id), (triplet_num, neighbor_num))
            negative_meta.append((false_left_connections, left_degrees, false_right_connections, right_degrees))
        
        return negative_meta
    
    def get_nn(self, support_meta):
        # build negative view of a given neighborhood by replacing entities
        left_connections, left_degrees, right_connections, right_degrees = support_meta
        false_left_connections, false_right_connections = left_connections.clone(), right_connections.clone()
        triplet_num, neighbor_num = false_left_connections.shape[0], false_left_connections.shape[1]
        false_left_connections[:, :, 1] = torch.randint(len(self.rel2id), self.num_symbols, (triplet_num, neighbor_num))
        false_right_connections[:, :, 1] = torch.randint(len(self.rel2id), self.num_symbols, (triplet_num, neighbor_num))
        
        return (false_left_connections, left_degrees, false_right_connections, right_degrees)

    def reload(self):
        if self.parameter['eval_ckpt'] is not None:
            state_dict_file = os.path.join(self.parameter['eval_ckpt'])
        else:
            state_dict_file = os.path.join(self.state_dir, 'state_dict')
        self.state_dict_file = state_dict_file
        self.logger.info('Reload state_dict from {}'.format(state_dict_file))
        print('reload state_dict from {}'.format(state_dict_file))
        state = torch.load(state_dict_file, map_location=self.device)
        if os.path.isfile(state_dict_file):
            self.hire.load_state_dict(state)
        else:
            raise RuntimeError('No state dict in {}!'.format(state_dict_file))

    def save_checkpoint(self, epoch):
        torch.save(self.hire.state_dict(), os.path.join(self.ckpt_dir, 'state_dict_' + str(epoch) + '.ckpt'))

    def del_checkpoint(self, epoch):
        path = os.path.join(self.ckpt_dir, 'state_dict_' + str(epoch) + '.ckpt')
        if os.path.exists(path):
            os.remove(path)
        else:
            raise RuntimeError('No such checkpoint to delete: {}'.format(path))

    def save_best_state_dict(self, best_epoch):
        shutil.copy(os.path.join(self.ckpt_dir, 'state_dict_' + str(best_epoch) + '.ckpt'),
                    os.path.join(self.state_dir, 'state_dict'))

    def write_training_log(self, data, epoch):
        self.loss_logger.info("Epoch: {}\tLoss_margin: {:.4f}\tLoss_info: {:.4f}\r".format(epoch, data['Loss_margin'], data['Loss_info']))

    def write_validating_log(self, data, epoch):
        self.loss_logger.info("Epoch: {}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(epoch, data['MRR'], data['Hits@10'], data['Hits@5'], data['Hits@1']))

    def logging_training_data(self, data, epoch):
        self.logger.info("Epoch: {}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
                      epoch, data['MRR'], data['Hits@10'], data['Hits@5'], data['Hits@1']))

    def logging_eval_data(self, data, state_path, istest=False):
        setname = 'dev set'
        if istest:
            setname = 'test set'
        self.logger.info("Eval {} on {}".format(state_path, setname))
        self.logger.info("MRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
                      data['MRR'], data['Hits@10'], data['Hits@5'], data['Hits@1']))
            
    def rank_predict(self, data, x, ranks):
        query_idx = x.shape[0] - 1
        _, idx = torch.sort(x, descending=True)
        rank = list(idx.cpu().numpy()).index(query_idx) + 1
        ranks.append(rank)
        if rank <= 10:
            data['Hits@10'] += 1
        if rank <= 5:
            data['Hits@5'] += 1
        if rank == 1:
            data['Hits@1'] += 1
        data['MRR'] += 1.0 / rank

    def do_one_step(self, task, iseval=False, curr_rel='', istest=False):
        loss, p_score, n_score = 0, 0, 0

        support_meta = self.get_task_meta([task[0]], iseval)[0]
        support_negative_meta = []
        for i in range(self.max_nn_meta):
            support_negative_meta.append(self.get_negative_meta(support_meta))
        if not iseval:
            self.optimizer.zero_grad()
            p_score, n_score, positive_context, negative_context, support_emb = self.hire(task, iseval, curr_rel, support_meta, support_negative_meta)
            negative_context = torch.stack(negative_context, dim=1)
            y = torch.Tensor([1]).to(self.device)
            loss_margin = self.hire.loss_func(p_score, n_score, y)
            loss_infonce = InfoNCE(negative_mode='paired')
            loss_info = loss_infonce(support_emb, positive_context, negative_context)
            loss_info = 0.05 * loss_info
            loss = loss_margin + loss_info
            loss.backward()
            self.optimizer.step()
        elif curr_rel != '':
            p_score, n_score, _, _, _ = self.hire(task, iseval, curr_rel, support_meta)
            y = torch.Tensor([1]).to(self.device)
            loss_margin = self.hire.loss_func(p_score, n_score, y)
            loss_info = 0
        return loss_margin, loss_info, p_score, n_score

    def train(self):
        best_epoch = 0
        best_value = 0
        bad_counts = 0

        for e in range(self.epoch):
            train_task, curr_rel = self.train_data_loader.next_batch()
            loss_margin, loss_info, _, _ = self.do_one_step(train_task, iseval=False, curr_rel=curr_rel, istest=False)
            if e % self.print_epoch == 0:
                loss_margin = loss_margin.item()
                loss_info = loss_info.item()
                self.write_training_log({'Loss_margin': loss_margin, 'Loss_info': loss_info}, e)
            if e % self.checkpoint_epoch == 0 and e != 0:
                print('Epoch  {} has finished, saving...'.format(e))
                self.save_checkpoint(e)
            if e % self.eval_epoch == 0 and e != 0:
                print('Epoch  {} has finished, validating...'.format(e))
                valid_data = self.eval(istest=False, epoch=e)
                self.write_validating_log(valid_data, e)
                metric = self.parameter['metric']
                # early stopping checking
                if valid_data[metric] > best_value:
                    best_value = valid_data[metric]
                    best_epoch = e
                    print('\tBest model | {0} of valid set is {1:.3f}'.format(metric, best_value))
                    bad_counts = 0
                    # save current best
                    self.save_checkpoint(best_epoch)
                else:
                    print('\tBest {0} of valid set is {1:.3f} at {2} | bad count is {3}'.format(
                        metric, best_value, best_epoch, bad_counts))
                    bad_counts += 1
                if bad_counts >= self.early_stopping_patience:
                    print('\tEarly stopping at epoch %d' % e)
                    break
        print('Training has finished')
        print('\tBest epoch is {0} | {1} of valid set is {2:.3f}'.format(best_epoch, metric, best_value))
        self.save_best_state_dict(best_epoch)
        print('Finish')

    def eval(self, istest=False, epoch=None):
        self.hire.eval()
        self.hire.rel_q_sharing = dict()
        if istest:
            data_loader = self.test_data_loader
        else:
            data_loader = self.dev_data_loader
        data_loader.curr_tri_idx = 0

        data = {'MRR': 0, 'Hits@1': 0, 'Hits@5': 0, 'Hits@10': 0}
        ranks = []
        t = 0
        temp = dict()
        while True:
            eval_task, curr_rel = data_loader.next_one_on_eval()
            if eval_task == 'EOT':
                break
            t += 1
            _, _, p_score, n_score = self.do_one_step(eval_task, iseval=True, curr_rel=curr_rel, istest=istest)
            x = torch.cat([n_score, p_score], 1).squeeze()
            self.rank_predict(data, x, ranks)
            for k in data.keys():
                temp[k] = data[k] / t
            sys.stdout.write("{}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
                t, temp['MRR'], temp['Hits@10'], temp['Hits@5'], temp['Hits@1']))
            sys.stdout.flush()
        for k in data.keys():
            data[k] = round(data[k] / t, 3)

        if self.parameter['step'] == 'train':
            self.logging_training_data(data, epoch)
        else:
            self.logging_eval_data(data, self.state_dict_file, istest)
        print("{}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
               t, data['MRR'], data['Hits@10'], data['Hits@5'], data['Hits@1']))
        self.hire.train()
        return data

    