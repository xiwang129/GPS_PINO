from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import torch
from torch.utils.data import DataLoader

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
    myloss = LpLoss(size_average=False)
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
   
    for i in range(2):
        data = next(iter(input))
        x = data[0]
        y = data[1]
        output = model(x).reshape(y.shape)
        output = output * mollifier
        pino_output = pino_model(x).reshape(y.shape)
        pino_output = pino_output * mollifier

        # ax = fig.add_subplot(3,3,i*3+1)
        # ax.imshow(x.squeeze())
        # if i == 0:
        #     ax.set_title('Input x')

        ax = fig.add_subplot(3,3,i*3+1)
        ax.imshow(y.squeeze())
        if i == 0:
            ax.set_title('Ground Truth')
        ax.set_xticks([])  
        ax.set_yticks([])
        
        ax = fig.add_subplot(3,3,i*3+2)
        ax.imshow(output.squeeze().detach().numpy())
        if i == 0:
            ax.set_title('Ours Prediction')
        ax.set_xticks([])  
        ax.set_yticks([])

        ax = fig.add_subplot(3,3,i*3+3)
        ax.imshow(pino_output.squeeze().detach().numpy())
        if i == 0:
            ax.set_title('PINO Prediction')
        ax.set_xticks([])  
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig('1000_darcy.png')
    fig.show()


def plot_burgers(t,x):
    # plot u(t=const, x) cross-sections
    fig = plt.figure(figsize=(7,4))
    t_cross_sections = [0.25, 0.5, 0.75]
    for i, t_cs in enumerate(t_cross_sections):
        plt.subplot(gs[1, i])
        tx = np.stack([np.full(t_flat.shape, t_cs), x_flat], axis=-1)
        u = network.predict(tx, batch_size=num_test_samples)
        plt.plot(x_flat, u)
        plt.title('t={}'.format(t_cs))
        plt.xlabel('x')
        plt.ylabel('u(t,x)')
    plt.tight_layout()
    plt.show()

    # gs = GridSpec(2, 3)
    # plt.subplot(gs[0, :])
    # plt.pcolormesh(t, x, u, cmap='rainbow')
    # plt.xlabel('t')
    # plt.ylabel('x')
    # cbar = plt.colorbar(pad=0.05, aspect=10)
    # cbar.set_label('u(t,x)')
    # cbar.mappable.set_clim(-1, 1)