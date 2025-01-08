from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from .losses import LpLoss, darcy_loss, PINO_loss,gps_darcy_loss,GPS_PINO_loss


def eval_darcy(model,
               dataloader,
               config,
               device,
               use_tqdm=True):
    model.eval()
    myloss = LpLoss(size_average=True)
    if use_tqdm:
        pbar = tqdm(dataloader, dynamic_ncols=True, smoothing=0.05)
    else:
        pbar = dataloader

    mesh = dataloader.dataset.mesh
    mollifier = torch.sin(np.pi * mesh[..., 0]) * torch.sin(np.pi * mesh[..., 1]) * 0.001
    mollifier = mollifier.to(device)
    f_val = []
    test_err = []

    with torch.no_grad():
        for x, y,z in pbar:
            x, y = x.to(device), y.to(device)

            pred = model(x).reshape(y.shape)
            pred = pred * mollifier

            data_loss = myloss(pred, y)
            a = x[..., 0]
            f_loss = darcy_loss(pred, a)
            # f_loss,gps_loss = gps_darcy_loss(pred,a)
            # total = f_loss.item() + gps_loss.item()

            test_err.append(data_loss.item())
            f_val.append(f_loss.item())
            # f_val.append(total)
            if use_tqdm:
                pbar.set_description(
                    (
                        f'Equation error: {f_loss.item():.5f}, test l2 error: {data_loss.item()}'
                    )
                )
    mean_f_err = np.mean(f_val)
    std_f_err = np.std(f_val, ddof=1) / np.sqrt(len(f_val))

    mean_err = np.mean(test_err)
    std_err = np.std(test_err, ddof=1) / np.sqrt(len(test_err))

    return mean_f_err,std_f_err,mean_err,std_err

    
def eval_burgers(model,
                 dataloader,
                 v,
                 config,
                 device,
                 use_tqdm=True):
    model.eval()
    myloss = LpLoss(size_average=True)
    if use_tqdm:
        pbar = tqdm(dataloader, dynamic_ncols=True, smoothing=0.05)
    else:
        pbar = dataloader

    test_err = []
    f_err = []

    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        out = model(x).reshape(y.shape)
        data_loss = myloss(out, y)

        loss_u, f_loss = PINO_loss(out, x[:, 0, :, 0], v)
        test_err.append(data_loss.item())
        f_err.append(f_loss.item())

    mean_f_err = np.mean(f_err)
    std_f_err = np.std(f_err, ddof=1) / np.sqrt(len(f_err))

    mean_err = np.mean(test_err)
    std_err = np.std(test_err, ddof=1) / np.sqrt(len(test_err))

    print(f'==Averaged relative L2 error mean: {mean_err}, std error: {std_err}==\n'
          f'==Averaged equation error mean: {mean_f_err}, std error: {std_f_err}==')
    
def get_molifier(mesh):
    mollifier = 0.001 * torch.sin(np.pi * mesh[..., 0]) * torch.sin(np.pi * mesh[..., 1])
    return mollifier

def plot_eval(model,pino_model,test_set):
    fig = plt.figure(figsize=(5,5))
    model.eval()
    pino_model.eval()
    
    input = DataLoader(test_set,batch_size=1)
    mollifier = get_molifier(input.dataset.mesh)
   
    for i in range(1):
        # data = next(iter(input))
        data = input.dataset[i]
        x = data[0].unsqueeze(0)
        y = data[1]
        output = model(x).reshape(y.shape)
        output = output * mollifier
        pino_output = pino_model(x).reshape(y.shape)
        pino_output = pino_output * mollifier

        ax = fig.add_subplot(2,3,1)
        ax.imshow(x[:,:,:,0].squeeze(0),cmap='gray')     
        ax.set_title('Input x',fontsize=10)
        ax.set_xticks([])  
        ax.set_yticks([])
      
        ax = fig.add_subplot(2,3,2)
        ax.imshow(output.squeeze().detach().numpy())    
        ax.set_title('Ours',fontsize=10)
        ax.set_xticks([])  
        ax.set_yticks([])

        ax = fig.add_subplot(2,3,3)
        ax.imshow(pino_output.squeeze().detach().numpy())    
        ax.set_title('PINO',fontsize=10)
        ax.set_xticks([])  
        ax.set_yticks([])

    ax = fig.add_subplot(1,3,1)
    ax.imshow(y.squeeze())
    ax.set_xlabel('Ground Truth',fontsize=10)
    # ax.set_xlabel('Ground Truth',fontsize=10)
    ax.set_xticks([])  
    ax.set_yticks([])
        
    ax = fig.add_subplot(1,3,2)
    cax = ax.imshow(output.squeeze().detach().numpy()-y.squeeze().detach().numpy())
    # cbar = fig.colorbar(cax, ax=ax) 
    ax.set_xlabel('Abs. Error',fontsize=10)
    ax.set_xticks([])  
    ax.set_yticks([])
    # cbar.set_ticks([])  

    ax = fig.add_subplot(1,3,3)
    cax = ax.imshow(pino_output.squeeze().detach().numpy()-y.squeeze().detach().numpy())
    # cbar = fig.colorbar(cax, ax=ax)    
    ax.set_xlabel('Abs. Error',fontsize=10)
    ax.set_xticks([])  
    ax.set_yticks([])
    # cbar.set_ticks([])  

    plt.tight_layout()     
    plt.savefig('100_darcy.png')
    fig.show()


def plot_burgers_error(key,config,model,test_loader):
    Nx = config['data']['nx']
    Nt = config['data']['nt'] + 1
    N = 100
    model.eval()
    test_x = np.zeros((N,Nt,Nx,3))
    preds_y = np.zeros((N,Nt,Nx))
    test_y = np.zeros((N,Nt,Nx))
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data_x, data_y = data
            pred_y = model(data_x).reshape(data_y.shape)
            test_x[i] = data_x.cpu().numpy()
            test_y[i] = data_y.cpu().numpy()
            preds_y[i] = pred_y.cpu().numpy()
    
    pred = preds_y[key]
    true = test_y[key]

    a = test_x[key]
    Nt, Nx, _ = a.shape
    u0 = a[0,:,0]
    T = a[:,:,2]
    X = a[:,:,1]
    x = X[0]

    fig, axes = plt.subplots(2, 3, figsize=(24,5))
    
    axes[0,0].plot(x, u0,figsize=(5,5))
    axes[0,0].set_xlabel('$x$')
    axes[0,0].set_ylabel('$u$')
    axes[0,0].set_title('Intial Condition $u(x)$')
    axes[0,0].set_xlim([0,1])
   
    # plt.pcolor(XX,TT, S_pred, cmap='jet')
    axes[0,1].pcolormesh(X, T, pred, cmap='jet', shading='gouraud')
    # plt.colorbar()
    axes[0,1].set_xlabel('$x$')
    axes[0,1].set_ylabel('$t$')
    axes[0,1].set_title(f'Ours $u(x,t)$')
    axes[0,1].axis('square')
  

    # plt.pcolor(XX,TT, S_pred, cmap='jet')
    axes[0,2].pcolormesh(X, T, pred, cmap='jet', shading='gouraud')
    # plt.colorbar()
    axes[0,2].set_xlabel('$x$')
    axes[0,2].set_ylabel('$t$')
    axes[0,2].set_title(f'PINO $u(x,t)$')
    axes[0,2].axis('square')
  
  
    # plt.pcolor(XX,TT, S_test, cmap='jet')
    axes[1,0].pcolormesh(X, T, true, cmap='jet', shading='gouraud')
    # plt.colorbar()
    axes[1,0].set_xlabel('$x$')
    axes[1,0].set_ylabel('$t$')
    axes[1,0].set_title(f'Exact $u(x,t)$')
    axes[0,3].axis('square')

 
    # plt.pcolor(XX,TT, S_pred - S_test, cmap='jet')
    axes[1,1].pcolormesh(X, T, pred - true, cmap='jet', shading='gouraud')
    # plt.colorbar()
    axes[1,1].set_xlabel('$x$')
    axes[1,1].set_ylabel('$t$')
    axes[1,1].set_title('Absolute error')
    axes[0,3].axis('square')

    # plt.pcolor(XX,TT, S_pred - S_test, cmap='jet')
    axes[1,2].pcolormesh(X, T, pred - true, cmap='jet', shading='gouraud')
    # plt.colorbar()
    axes[1,2].set_xlabel('$x$')
    axes[1,2].set_ylabel('$t$')
    axes[1,2].set_title('Absolute error')
    axes[0,3].axis('square')

    plt.tight_layout()
    plt.savefig('100_burgers.png')
    fig.show()


def plot_reso():

    x = np.array([42,61,211])   
    gps = np.array([0.87,1.20,6.08,])              
    pino = np.array([1.08,1.51,7.36])
 
    fig,ax1 = plt.subplots(figsize=(5,5))
    ax1.plot(x, gps, label="Ours", color='blue', linestyle='-',marker='o', linewidth=2)  
    ax1.plot(x, pino, label="PINO", color='green', linestyle='--', marker='o',linewidth=2)  

    ax1.set_xlabel("Resolution", fontsize=20)
    ax1.set_ylabel("Mean Equation Error", fontsize=20)
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)

    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('reso_darcy.png')  
    plt.show()

def plot_reso_l2():
   
    x = np.array([42,61,211])   
    gps = np.array([0.045,0.046,0.046]) 
    pino = np.array([0.066,0.066,0.068])             
               
    fig,ax1 = plt.subplots(figsize=(5,5))
    ax1.plot(x, gps, label="Ours", color='blue', linestyle='-',marker='o', linewidth=2) 
    ax1.plot(x, pino, label="PINO", color='green', linestyle='--', marker='o',linewidth=2)  

    ax1.set_xlabel("Resolution", fontsize=20)
    ax1.set_ylabel("Mean L2 loss", fontsize=20)
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)

    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('reso_l2_darcy.png')  
    plt.show()


def plot_dim():
    x = ['v1','v1+v2','v1+v2+v3','v1+v2+v3+v4','All']  
    error = np.array([0.078,0.077,0.070,0.071,0.068]) 

    pino = np.array([0.095,0.095,0.095,0.095,0.095])
    ours = np.array([0.068,0.068,0.068,0.068,0.068])

    fig,ax1 = plt.subplots(figsize=(5,5))
  
    ax1.plot(x,error, color='b', marker='x',label='Vectors')
    ax1.plot(x, pino, color='r', marker='o', label='PINO')
    ax1.plot(x, ours, color='g', marker='o', label='Ours')

    ax1.set_xlabel('Prolongation Vectors',fontsize=20)
    ax1.set_ylabel('Mean Equation Error',fontsize=20)
    ax1.set_xticklabels(x, rotation=25)
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)

    ax1.legend(fontsize=12) 
    plt.tight_layout()
    plt.savefig('vs_burgers.png')
    plt.show() 