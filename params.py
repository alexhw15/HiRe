import torch
import argparse


def get_params():
    args = argparse.ArgumentParser()
    args.add_argument("-data", "--dataset", default="NELL-One", choices=["NELL-One", "Wiki-One"])
    args.add_argument("-path", "--data_path", default="./Nell-HiRe", type=str) 
    args.add_argument("-seed", "--seed", default=1520, type=int)
    args.add_argument("-few", "--few", default=1, type=int)
    args.add_argument("-nq", "--num_query", default=3, type=int)
    args.add_argument("-metric", "--metric", default="MRR", choices=["MRR", "Hits@10", "Hits@5", "Hits@1"])

    args.add_argument("-dim", "--embed_dim", default=100, type=int)
    args.add_argument("-bs", "--batch_size", default=1024, type=int)
    args.add_argument("-lr", "--learning_rate", default=0.001, type=float)
    args.add_argument("-es_p", "--early_stopping_patience", default=30, type=int)

    args.add_argument("-epo", "--epoch", default=100000, type=int)
    args.add_argument("-prt_epo", "--print_epoch", default=100, type=int)
    args.add_argument("-eval_epo", "--eval_epoch", default=1000, type=int)
    args.add_argument("-ckpt_epo", "--checkpoint_epoch", default=1000, type=int)

    args.add_argument("-b", "--beta", default=5, type=float)
    args.add_argument("-m", "--margin", default=1, type=float)
    args.add_argument("-p", "--dropout_p", default=0.5, type=float)
    
    args.add_argument("-gpu", "--device", default=0, type=int)
    args.add_argument("-prefix", "--prefix", default="exp1", type=str)
    args.add_argument("-step", "--step", default="train", type=str, choices=['train', 'test', 'dev'])
    args.add_argument("-log_dir", "--log_dir", default="log", type=str)
    args.add_argument("-state_dir", "--state_dir", default="./Hire-state", type=str)
    args.add_argument("-eval_ckpt", "--eval_ckpt", default=None, type=str)
    args.add_argument("-eval_by_rel", "--eval_by_rel", default=False, type=bool)
    args.add_argument("-max_neighbor", "--max_neighbor", default=50, type=int)
    args.add_argument("-max_nn_meta", "--max_nn_meta", default=1, type=int)
    args.add_argument("-embed_model", "--embed_model", default="TransE", type=str)

    args = args.parse_args()
    params = {}
    for k, v in vars(args).items():
        params[k] = v

    if args.dataset == 'NELL-One':
        params['embed_dim'] = 100
    elif args.dataset == 'Wiki-One':
        params['embed_dim'] = 50

    params['device'] = torch.device('cuda:'+str(args.device))
    params['prefix'] = params['prefix'] + '-shot' + str(params['few']) + '-lr' + str(params['learning_rate'])

    return params


data_dir = {
    'train_tasks': '/train_tasks.json',
    'test_tasks': '/test_tasks.json',
    'dev_tasks': '/dev_tasks.json',
    'rel2candidates': '/rel2candidates.json',
    'e1rel_e2': '/e1rel_e2.json',
    'ent2ids': '/ent2ids',
    'relation2ids': '/rel2ids-transd'
}
