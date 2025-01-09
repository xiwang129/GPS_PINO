import numpy as np
import torch
import torch.nn.functional as F


def FDM_Darcy(u, a, D=1):
    batchsize = u.size(0)
    size = u.size(1)
    u = u.reshape(batchsize, size, size)
    a = a.reshape(batchsize, size, size)
    dx = D / (size - 1)
    dy = dx

    # ux: (batch, size-2, size-2)
    ux = (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / (2 * dx)
    uy = (u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) / (2 * dy)

    a = a[:, 1:-1, 1:-1]

    aux = a * ux
    auy = a * uy
    auxx = (aux[:, 2:, 1:-1] - aux[:, :-2, 1:-1]) / (2 * dx)
    auyy = (auy[:, 1:-1, 2:] - auy[:, 1:-1, :-2]) / (2 * dy)
    Du = - (auxx + auyy)
    return Du


def GPS_FDM_Darcy(u, a, D=1, f=None):
    if (f is None):
        batchsize = u.size(0)
        size = u.size(1)
        u = u.reshape(batchsize, size, size)
        a = a.reshape(batchsize, size, size)
        dx = D / (size - 1)
        dy = dx

        # ux: (batch, size-2, size-2)
        ux = (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / (2 * dx)
        uy = (u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) / (2 * dy)

        a = a[:, 1:-1, 1:-1]
        
        aux = a * ux
        auy = a * uy
        aux_x = (aux[:, 2:, 1:-1] - aux[:, :-2, 1:-1]) / (2 * dx)
        auy_y = (auy[:, 1:-1, 2:] - auy[:, 1:-1, :-2]) / (2 * dy)
        Du = - (aux_x + auy_y)
        
        #### the original loss is -[(ku_x)_x + (ku_y)_y] + f = 0
        
        #### our loss is -D_x[ [(ku_x)_x + (ku_y)_y] + f = 0 and -D_y[ [(ku_x)_x + (ku_y)_y] + f = 0
        #### our loss is -[(ku_x)_xx + (ku_y)_yx] = 0
        #### our loss is -[(ku_x)_xy + (ku_y)_yy] = 0
        
        aux_xx = (aux_x[:, 2:, 1:-1] - aux_x[:, :-2, 1:-1]) / (2 * dx)
        auy_yx = (auy_y[:, 2:, 1:-1] - auy_y[:, :-2, 1:-1]) / (2 * dx)
        
        aux_xy = (aux_x[:, 1:-1, 2:] - aux_x[:, 1:-1, :-2]) / (2 * dy)
        auy_yy = (auy_y[:, 1:-1, 2:] - auy_y[:, 1:-1, :-2]) / (2 * dy)
        
        Dx_Du = - (aux_xx + auy_yx)
        Dy_Du = - (aux_xy + auy_yy)
        
    return Du, Dx_Du, Dy_Du

def gps_darcy_loss(u, a):
    batchsize = u.size(0)
    size = u.size(1)
    u = u.reshape(batchsize, size, size)
    a = a.reshape(batchsize, size, size)
    lploss = LpLoss(size_average=True)
   
    Du, Dx_Du, Dy_Du = GPS_FDM_Darcy(u, a)
    f = torch.ones(Du.shape, device=u.device)
    loss_f = lploss.rel(Du, f)

    # loss_gps = lploss.rel(Dx_Du, torch.zeros(Dx_Du.shape, device=u.device)) + lploss.rel(Dy_Du, torch.zeros(Dy_Du.shape, device=u.device))
    loss_gps = lploss.abs(Dx_Du, torch.zeros(Dx_Du.shape, device=u.device)) + lploss.abs(Dy_Du, torch.zeros(Dy_Du.shape, device=u.device))

    return loss_f,loss_gps

def darcy_loss(u, a):
    batchsize = u.size(0)
    size = u.size(1)
    u = u.reshape(batchsize, size, size)
    a = a.reshape(batchsize, size, size)
    lploss = LpLoss(size_average=True)
 
    Du = FDM_Darcy(u, a)
    f = torch.ones(Du.shape, device=u.device)
    loss_f = lploss.rel(Du, f)
    return loss_f


class LpLoss(object):
    '''
    loss function with rel/abs Lp loss
    '''
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


def FDM_Burgers(u, v, D=1):
    batchsize = u.size(0)
    nt = u.size(1)
    nx = u.size(2)

    u = u.reshape(batchsize, nt, nx)
    dt = D / (nt-1)
    dx = D / (nx)

    u_h = torch.fft.fft(u, dim=2)
    # Wavenumbers in y-direction
    k_max = nx//2
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=u.device),
                     torch.arange(start=-k_max, end=0, step=1, device=u.device)), 0).reshape(1,1,nx)
    ux_h = 2j *np.pi*k_x*u_h
    uxx_h = 2j *np.pi*k_x*ux_h
    ux = torch.fft.irfft(ux_h[:, :, :k_max+1], dim=2, n=nx)
    uxx = torch.fft.irfft(uxx_h[:, :, :k_max+1], dim=2, n=nx)
    ut = (u[:, 2:, :] - u[:, :-2, :]) / (2 * dt)
    Du = ut + (ux*u - v*uxx)[:,1:-1,:]
    return Du


def GPS_FDM_Burgers(u, v, D=1):
    batchsize = u.size(0)
    nt = u.size(1)
    nx = u.size(2)

    u = u.reshape(batchsize, nt, nx)
    dt = D / (nt-1)
    dx = D / (nx)

    u_h = torch.fft.fft(u, dim=2)
    # Wavenumbers in y-direction
    k_max = nx//2
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=u.device),
                     torch.arange(start=-k_max, end=0, step=1, device=u.device)), 0).reshape(1,1,nx)
    ux_h = 2j *np.pi*k_x*u_h
    uxx_h = 2j *np.pi*k_x*ux_h
    uxxx_h = 2j *np.pi*k_x*uxx_h
    
    
    u_x = torch.fft.irfft(ux_h[:, :, :k_max+1], dim=2, n=nx)
    u_xx = torch.fft.irfft(uxx_h[:, :, :k_max+1], dim=2, n=nx)
    
    u_xxx = torch.fft.irfft(uxxx_h[:, :, :k_max+1], dim=2, n=nx)
    
    u_t = (u[:, 2:, :] - u[:, :-2, :]) / (2 * dt)
    
    
    u_xt = (u_x[:, 2:, :] - u_x[:, :-2, :]) / (2 * dt)
    u_xxt = (u_xx[:, 2:, :] - u_xx[:, :-2, :]) / (2 * dt)
    u_tt = (u_t[:, 2:, :] - u_t[:, :-2, :]) / (2 * dt)
    
    
    Du = u_t + (u_x*u - v*u_xx)[:,1:-1,:] 
    Dx_Du = u_xt + ((u_x)**2 + u*u_xx - v*u_xxx)[:,1:-1,:]            
    Dt_Du = u_tt + ((u_x[:, 1:-1, :]  * u_t) + u[:, 1:-1, :]*u_xt - v*u_xxt)[:,1:-1,:]   
    return Du, Dx_Du, Dt_Du


def PINO_loss(u, u0, v):
    batchsize = u.size(0)
    nt = u.size(1)
    nx = u.size(2)

    u = u.reshape(batchsize, nt, nx)
    # lploss = LpLoss(size_average=True)

    index_t = torch.zeros(nx,).long()
    index_x = torch.tensor(range(nx)).long()
    boundary_u = u[:, index_t, index_x]
    loss_u = F.mse_loss(boundary_u, u0)

    Du = FDM_Burgers(u, v)[:, :, :]
    f = torch.zeros(Du.shape, device=u.device)
    loss_f = F.mse_loss(Du, f)
    return loss_u, loss_f


def GPS_PINO_loss(u, u0, v):
    batchsize = u.size(0)
    nt = u.size(1)
    nx = u.size(2)

    u = u.reshape(batchsize, nt, nx)
    # lploss = LpLoss(size_average=True)

    index_t = torch.zeros(nx,).long()
    index_x = torch.tensor(range(nx)).long()
    boundary_u = u[:, index_t, index_x]
    loss_u = F.mse_loss(boundary_u, u0)

    # Du, Dx_Du, Dt_Du = GPS_FDM_Burgers(u, v)[:, :, :]
    Du, Dx_Du, Dt_Du = GPS_FDM_Burgers(u, v)
    f = torch.zeros(Du.shape, device=u.device)
    loss_f = F.mse_loss(Du, f)
    
    t = torch.linspace(0, 1, nt).reshape(1, nt, 1).repeat(batchsize, 1, nx).float()
    x = torch.linspace(0, 1, nx).reshape(1, 1, nx).repeat(batchsize, nt, 1).float()
    
    def crop(x):
        return x[:, 1:-1, :]
    
    v1Du = -Dx_Du
    v2Du = -Dt_Du
    v3Du = -(crop(3*Du+crop(x)*Dx_Du) + 2*crop(crop(t))*Dt_Du)             
    v4Du = -crop(t)*Dx_Du
    v5Du = -crop(crop(t))*(crop(3*Du+crop(x)*Dx_Du) + crop(crop(t))*Dt_Du)  
    
    loss_gps = (F.mse_loss(v1Du, torch.zeros(v1Du.shape, device=u.device)) +  
                F.mse_loss(v2Du, torch.zeros(v2Du.shape, device=u.device)) + 
                F.mse_loss(v3Du, torch.zeros(v3Du.shape, device=u.device)) + 
                F.mse_loss(v4Du, torch.zeros(v4Du.shape, device=u.device)) + 
                F.mse_loss(v5Du, torch.zeros(v5Du.shape, device=u.device)))

    return loss_u, loss_f, loss_gps

def Sym_PINO_loss(u, u0, v,device):
    batchsize = u.size(0)
    nt = u.size(1)
    nx = u.size(2)

    u = u.reshape(batchsize, nt, nx)
    # lploss = LpLoss(size_average=True)

    index_t = torch.zeros(nx,).long()
    index_x = torch.tensor(range(nx)).long()
    boundary_u = u[:, index_t, index_x]
    loss_u = F.mse_loss(boundary_u, u0)

    Du, Dx_Du, Dt_Du = GPS_FDM_Burgers(u, v)
    f = torch.zeros(Du.shape, device=u.device)
    loss_f = F.mse_loss(Du, f)

    t = torch.linspace(0, 1, nt).reshape(1, nt, 1).repeat(batchsize, 1, nx).float().to(device)
    x = torch.linspace(0, 1, nx).reshape(1, 1, nx).repeat(batchsize, nt, 1).float().to(device)
    
    def crop(x):
        return x[:, 1:-1, :].to(device)
    
    # v3Du = 2*crop(crop(t))* Dt_Du + crop(crop(x)*Dx_Du- 3*Du)   
    # v5Du = crop(crop(t))*(crop(crop(t))*Dt_Du + crop(crop(x)*Dx_Du - 3*Du))
  
    # loss_gps = (F.mse_loss(v3Du, torch.zeros(v3Du.shape, device=u.device)) + F.mse_loss(v5Du, torch.zeros(v5Du.shape, device=u.device)))
    
    v5Du = crop(t) * Du
    loss_gps = F.mse_loss(v5Du,torch.zeros(v5Du.shape, device=u.device))

    return loss_u, loss_f, loss_gps
  