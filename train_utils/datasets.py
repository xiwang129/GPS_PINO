import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset
from .utils import  torch2dgrid


class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        self.data = scipy.io.loadmat(self.file_path)
        self.old_mat = True

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float


class BurgersLoader(object):
    def __init__(self, x_data, y_data, nx=128, nt=100, sub=1, sub_t=1, new=True):
        self.sub = sub
        self.sub_t = sub_t
        self.s = 128 // sub 
        self.T = nt // sub_t
        self.new = new
        if new:
            self.T += 1
        
        self.x_data = x_data[:,::sub]
        self.y_data  = y_data[:,::sub_t,::sub] 
      
    def make_loader(self, n_sample, batch_size, start=0, train=True):
        Xs = self.x_data[start:start + n_sample]
        ys = self.y_data[start:start + n_sample]

        if self.new:
            gridx = torch.tensor(np.linspace(0, 1, self.s + 1)[:-1], dtype=torch.float)
            gridt = torch.tensor(np.linspace(0, 1, self.T), dtype=torch.float)
        else:
            gridx = torch.tensor(np.linspace(0, 1, self.s), dtype=torch.float)
            gridt = torch.tensor(np.linspace(0, 1, self.T + 1)[1:], dtype=torch.float)
        gridx = gridx.reshape(1, 1, self.s)
        gridt = gridt.reshape(1, self.T, 1)

        # Xs = Xs.reshape(n_sample, 1, self.s).repeat([1, self.T, 1])
        Xs = Xs.unsqueeze(1).repeat([1, self.T, 1])
        Xs = torch.stack([Xs, gridx.repeat([n_sample, self.T, 1]), gridt.repeat([n_sample, 1, self.s])], dim=3)
        dataset = torch.utils.data.TensorDataset(Xs, ys)
        if train:
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        else:
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return loader

class DarcyFlow(Dataset):
    def __init__(self,
                 datapath,
                 nx, sub,
                 offset=0,
                 num=1):
        self.S = int(nx // sub) + 1 if sub > 1 else nx
        data = scipy.io.loadmat(datapath)
        a = data['coeff']
        u = data['sol']
        self.a = torch.tensor(a[offset: offset + num, ::sub, ::sub], dtype=torch.float)
        self.u = torch.tensor(u[offset: offset + num, ::sub, ::sub], dtype=torch.float)
        self.mesh = torch2dgrid(self.S, self.S)

    def __len__(self):
        return self.a.shape[0]

    def __getitem__(self, item):
        fa = self.a[item]
        return torch.cat([fa.unsqueeze(2), self.mesh], dim=2), self.u[item]


class DarcyIC(Dataset):
    def __init__(self,
                 datapath,
                 nx, sub,
                 offset=0,
                 num=1):
        self.S = int(nx // sub) + 1 if sub > 1 else nx
        data = scipy.io.loadmat(datapath)
        a = data['coeff']
        self.a = torch.tensor(a[offset: offset + num, ::sub, ::sub], dtype=torch.float)
        self.mesh = torch2dgrid(self.S, self.S)
        data = scipy.io.loadmat(datapath)
        a = data['coeff']
        u = data['sol']
        self.a = torch.tensor(a[offset: offset + num, ::sub, ::sub], dtype=torch.float)
        self.u = torch.tensor(u[offset: offset + num, ::sub, ::sub], dtype=torch.float)
        self.mesh = torch2dgrid(self.S, self.S)

    def __len__(self):
        return self.a.shape[0]

    def __getitem__(self, item):
        fa = self.a[item]
        return torch.cat([fa.unsqueeze(2), self.mesh], dim=2) 


class DarcyCombo(Dataset):
    def __init__(self, 
                 datapath, 
                 nx, 
                 sub, pde_sub, 
                 num=1000, offset=0) -> None:
        super().__init__()
        self.S = int(nx // sub) + 1 if sub > 1 else nx
        self.pde_S = int(nx // pde_sub) + 1 if sub > 1 else nx
        data = scipy.io.loadmat(datapath)
        a = data['coeff']
        u = data['sol']
        self.a = torch.tensor(a[offset: offset + num, ::sub, ::sub], dtype=torch.float)
        self.u = torch.tensor(u[offset: offset + num, ::sub, ::sub], dtype=torch.float)
        self.mesh = torch2dgrid(self.S, self.S)
        self.pde_a = torch.tensor(a[offset: offset + num, ::pde_sub, ::pde_sub], dtype=torch.float)
        self.pde_mesh = torch2dgrid(self.pde_S, self.pde_S)

    def __len__(self):
        return self.a.shape[0]

    def __getitem__(self, item):
        fa = self.a[item]
        pde_a = self.pde_a[item]
        data_ic = torch.cat([fa.unsqueeze(2), self.mesh], dim=2)
        pde_ic = torch.cat([pde_a.unsqueeze(2), self.pde_mesh], dim=2)
        return data_ic, self.u[item], pde_ic
