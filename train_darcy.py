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

from train_utils.losses import LpLoss, darcy_loss, gps_darcy_loss
from train_utils.datasets import DarcyCombo,DarcyFlow, DarcyIC, sample_data
from train_utils.utils import save_ckpt, count_params, dict2str
from train_utils.eval_2d import plot_eval,eval_darcy


def get_molifier(mesh, device):
    mollifier = 0.001 * torch.sin(np.pi * mesh[..., 0]) * torch.sin(np.pi * mesh[..., 1])
    return mollifier.to(device)


# @torch.no_grad()
# def eval_darcy(model, val_loader, criterion, 
#                device='cpu'):
#     mollifier = get_molifier(val_loader.dataset.mesh, device)
#     model.eval()
#     val_err = []
#     for a, u in val_loader:
#         a, u = a.to(device), u.to(device)
#         out = model(a).squeeze(dim=-1)
#         out = out * mollifier
#         val_loss = criterion(out, u)
#         val_err.append(val_loss.item())
#     N = len(val_loader)

#     avg_err = np.mean(val_err)
#     std_err = np.std(val_err, ddof=1) / np.sqrt(N)
#     return avg_err, std_err


def train(model, 
          train_u_loader,        # training data
          ic_loader,             # loader for initial conditions
        #   val_loader,            # validation data
          optimizer, 
          scheduler,
          device, config, args):
    save_step = config['train']['save_step']
    eval_step = config['train']['eval_step']

    f_weight = config['train']['f_loss']
    xy_weight = config['train']['xy_loss']
    gps_weight = config['train']['gps_loss']

    # set up directory
    base_dir = os.path.join('exp', config['log']['logdir'])
    ckpt_dir = os.path.join(base_dir, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)

    # loss fn
    lploss = LpLoss(size_average=True)
    # mollifier
    u_mol = get_molifier(train_u_loader.dataset.mesh, device)
    ic_mol = get_molifier(ic_loader.dataset.mesh, device)

    pbar = range(config['train']['num_iter'])
    pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.2)

    u_loader = sample_data(train_u_loader)
    ic_loader = sample_data(ic_loader)
    for e in pbar:
        log_dict = {}

        optimizer.zero_grad()
        # data loss
        if xy_weight > 0:
            ic, u = next(u_loader)
            u = u.to(device)
            ic = ic.to(device)
            out = model(ic).squeeze(dim=-1)
            out = out * u_mol
            data_loss = lploss(out, u)
        else:
            data_loss = torch.zeros(1, device=device)
        
        if f_weight > 0:
            # pde loss
            ic = next(ic_loader)
            ic = ic.to(device)
            out = model(ic).squeeze(dim=-1)
            out = out * ic_mol
            u0 = ic[..., 0]
            f_loss = darcy_loss(out, u0)
            f_loss, gps_loss = gps_darcy_loss(out, u0)
            log_dict['PDE'] = f_loss.item()
            log_dict['GPS'] = gps_loss.item()  ##### TODO: add to log dict
        else:
            f_loss = 0.0

        loss = data_loss * xy_weight + f_loss * f_weight + gps_loss * gps_weight #### TODO: add gps_weight to be something

        loss.backward()
        optimizer.step()
        scheduler.step()

        log_dict['train loss'] = loss.item()
        log_dict['data'] = data_loss.item()
        # if e % eval_step == 0:
        #     eval_err, std_err = eval_darcy(model, val_loader, lploss, device)
        #     log_dict['val error'] = eval_err
        logstr = dict2str(log_dict)
        pbar.set_description(
            (
                logstr
            )
        )
        
        if e % save_step == 0 and e > 0:
            ckpt_path = os.path.join(ckpt_dir, f'model-{e}.pt')
            save_ckpt(ckpt_path, model, optimizer, scheduler)

def subprocess(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # set random seed
    # config['seed'] = args.seed
    # seed = args.seed
    # torch.manual_seed(seed)
    # random.seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)

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
        # testset = DarcyFlow(datapath=config['test']['datapath'], 
        #                     nx=config['test']['nx'], 
        #                     sub=config['test']['sub'], 
        #                     offset=config['test']['offset'], 
        #                     num=config['test']['n_sample'])
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
        # test_err, std_err = eval_darcy(model, testloader, criterion, device)
        # print(f'Averaged test relative L2 error: {test_err}; Standard error: {std_err}')

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
        train(model, u_loader,ic_loader, # val_loader, 
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


#Averaged test relative L2 error: 0.9999838058948517; Standard error: 6.409100613427627e-08

#Averaged test relative L2 error: 0.19530367155373096; Standard error: 0.0046897652609006486
#Equation error: 2.66199, test l2 error
# ==Averaged equation error mean: 2.189151368260384, std error: 0.03880616839849416==
# ==Averaged relative L2  error mean: 0.19530367267876864, std error: 0.004689765254887614==

#Averaged test relative L2 error: 0.999987062573433; Standard error: 5.112165656527389e-08

#pino-1000
# Equation error: 0.89285, test l2 error: 0.0314990691840
# ==Averaged relative L2 error mean: 0.013370322527363896, std error: 0.00030735290672945985==
# ==Averaged equation error mean: 0.5832822696864605, std error: 0.007889941878350803==

#pino-100
#Equation error: 2.04474, test l2 error: 0.15833085775
# ==Averaged relative L2 error mean: 0.0666941185593605, std error: 0.0014719520411472333==
# ==Averaged equation error mean: 1.5508374849557875, std error: 0.017028828746709267==
# Done!

#pino-500
#Equation error: 1.04377, test l2 error: 
# ==Averaged relative L2 error mean: 0.01790224549919367, std error: 0.00045142157316416004==
# ==Averaged equation error mean: 0.7230256846547127, std error: 0.009276560356489797==

#gps-0.01-1000
# ==Averaged relative L2 error mean: 0.011363054542802275, std error: 0.00021840036498525203==
# ==Averaged equation error mean: 0.4907461925148964, std error: 0.00685823755522582==