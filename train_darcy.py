import os
import yaml
import random
from argparse import ArgumentParser
from tqdm import tqdm

import numpy as np
import torch

from torch.optim import Adam
from torch.utils.data import DataLoader

from models import FNO2d
from train_utils.train_2d import train_2d_operator

from train_utils.losses import LpLoss, darcy_loss, gps_darcy_loss
from train_utils.datasets import DarcyCombo,DarcyFlow, DarcyIC, sample_data
from train_utils.utils import save_ckpt, count_params, dict2str
from train_utils.eval_2d import plot_eval,eval_darcy


def get_molifier(mesh, device):
    mollifier = 0.001 * torch.sin(np.pi * mesh[..., 0]) * torch.sin(np.pi * mesh[..., 1])
    return mollifier.to(device)

def train_darcy(args, config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_config = config['data']

    # dataset = DarcyFlow(data_config['datapath'],
                        # nx=data_config['nx'], sub=data_config['sub'],
                        # offset=data_config['offset'], num=data_config['n_sample'])

    dataset = DarcyCombo(datapath=data_config['datapath'], 
                         nx=data_config['nx'], 
                         sub=data_config['sub'], 
                         pde_sub=data_config['pde_sub'], 
                         num=data_config['n_samples'], 
                         offset=data_config['offset'])
    train_loader = DataLoader(dataset, batch_size=config['train']['batchsize'], shuffle=True)
    model = FNO2d(modes1=config['model']['modes1'],
                  modes2=config['model']['modes2'],
                  fc_dim=config['model']['fc_dim'],
                  layers=config['model']['layers'],
                  act=config['model']['act'], 
                  pad_ratio=config['model']['pad_ratio']).to(device)
    # Load from checkpoint
    if 'ckpt' in config['train']:
        ckpt_path = config['train']['ckpt']
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % ckpt_path)

    optimizer = Adam(model.parameters(), betas=(0.9, 0.999),
                     lr=config['train']['base_lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=config['train']['milestones'],
                                                     gamma=config['train']['scheduler_gamma'])
    train_2d_operator(model,
                      train_loader,
                      optimizer, scheduler,
                      config, rank=0, log=args.log,
                      project=config['log']['project'],
                      group=config['log']['group'])


def subprocess(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # set random seed
    config['seed'] = args.seed
    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # create model 
    model = FNO2d(modes1=config['model']['modes1'],
                  modes2=config['model']['modes2'],
                  fc_dim=config['model']['fc_dim'],
                  layers=config['model']['layers'], 
                  act=config['model']['act'], 
                  pad_ratio=config['model']['pad_ratio']).to(device)
    num_params = count_params(model)
    config['num_params'] = num_params
    print(f'Number of parameters: {num_params}')

    # Load from checkpoint
    if args.ckpt:
        ckpt_path = args.ckpt
        ckpt = torch.load(ckpt_path,map_location=torch.device('cpu'))
        model.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % ckpt_path)
    
    if args.test:
        batchsize = config['test']['batchsize']
        testset = DarcyCombo(datapath=config['test']['datapath'], 
                            nx=config['test']['nx'], 
                            sub=config['test']['sub'], 
                            offset=config['test']['offset'], 
                            num=config['test']['n_sample'],
                            pde_sub=2)
        testloader = DataLoader(testset, batch_size=batchsize, num_workers=4)
        criterion = LpLoss()

        pino_model = FNO2d(modes1=config['model']['modes1'],
                  modes2=config['model']['modes2'],
                  fc_dim=config['model']['fc_dim'],
                  layers=config['model']['layers'], 
                  act=config['model']['act'], 
                  pad_ratio=config['model']['pad_ratio']).to(device)
        pino_path = 'darcy-1000-pino_new.pt'
        pino = torch.load(pino_path,map_location=torch.device('cpu'))
        pino_model.load_state_dict(pino['model'])
       
        plot_eval(model,pino_model,testset)
        test_err, std_err = eval_darcy(model, testloader, criterion, device)
        print(f'Averaged test relative L2 error: {test_err}; Standard error: {std_err}')

        mean_f_err,std_f_err,mean_err,std_err, =eval_darcy(model,testloader,config,device,use_tqdm=True)
        print(f'==Averaged relative L2 error mean: {mean_err}, std error: {std_err}==\n'
          f'==Averaged equation error mean: {mean_f_err}, std error: {std_f_err}==')
       
    else:
        # training set
        batchsize = config['train']['batchsize']
        u_set = DarcyFlow(datapath=config['data']['datapath'], 
                          nx=config['data']['nx'], 
                          sub=config['data']['sub'], 
                          offset=config['data']['offset'], 
                          num=config['data']['n_sample'])
        u_loader = DataLoader(u_set, batch_size=batchsize, num_workers=4, shuffle=True)
        ic_set = DarcyIC(datapath=config['data']['datapath'], 
                         nx=config['data']['nx'], 
                         sub=config['data']['pde_sub'], 
                         offset=config['data']['offset'], 
                         num=config['data']['n_sample'])
        ic_loader = DataLoader(ic_set, batch_size=batchsize, num_workers=4, shuffle=True)

        if 'ckpt' in config['train']:
            ckpt_path = config['train']['ckpt']
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt['model'])
            print('Weights loaded from %s' % ckpt_path)
        
        optimizer = Adam(model.parameters(), lr=config['train']['base_lr'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                         milestones=config['train']['milestones'], 
                                                         gamma=config['train']['scheduler_gamma'])
        train_darcy(model, u_loader,ic_loader, # val_loader, 
              optimizer, scheduler, 
              device, config, args)
                
    print('Done!')
        

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    # parse options
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config', type=str, default='./configs/darcy_flow.yaml')
    parser.add_argument('--log', action='store_true', help='Turn on the wandb')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--ckpt', type=str, default='darcy-1000-gps_001new.pt')
    parser.add_argument('--test', action='store_true', help='Test',default=True)
    args = parser.parse_args()
    if args.seed is None:
        args.seed = random.randint(0, 100000)

    subprocess(args)
