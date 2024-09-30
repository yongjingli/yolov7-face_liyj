import os
import torch
import subprocess
from abc import ABC


class GpuMemoryCalculator():
    def __init__(self, device="cuda:0"):
        self.device = device
        self.gpu_usage_0 = None
        self.m_allocated_0 = None
        self.m_reserved_0 = None
        self.m_max_reserved_0 = None

    def get_torch_memory_allocated(self):
        m_allocated = torch.cuda.memory_allocated(self.device) / 1024 ** 2  # MB
        m_allocated = round(m_allocated, 2)
        return m_allocated

    def get_torch_memory_reserved(self):
        m_reserved = torch.cuda.memory_reserved(self.device) / 1024 ** 2   # MB
        m_reserved = round(m_reserved, 2)
        return m_reserved

    def get_torch_max_memory_reserved(self):
        m_max_reserved = torch.cuda.max_memory_reserved(self.device) / 1024 ** 2   # MB
        m_max_reserved = round(m_max_reserved, 2)
        return m_max_reserved

    def get_gpu_memory_usage(self):
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used', '--format=csv,nounits,noheader'])

        # 解析输出, 获取每个GPU的显存占用
        gpu_memory_usage = {}
        for line in output.decode().strip().split('\n'):
            gpu_uuid, used_memory = line.split(', ')
            gpu_memory_usage[gpu_uuid] = float(used_memory)
        return gpu_memory_usage

    def get_device_gpu_memory_usage(self):
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used', '--format=csv,nounits,noheader'])

        # 解析输出, 获取每个GPU的显存占用
        gpu_memory_usage = {}
        for line in output.decode().strip().split('\n'):
            gpu_uuid, used_memory = line.split(', ')
            gpu_memory_usage[gpu_uuid] = float(used_memory)

        device_i = int(self.device.split(":")[1])
        device_gpu_memory_usage = gpu_memory_usage[list(gpu_memory_usage.keys())[device_i]]

        return device_gpu_memory_usage

    def get_torch_tensor_size(self, in_tensor):
        element_size = in_tensor.element_size()
        num_elements = in_tensor.nelement()
        in_tensor_size = element_size * num_elements
        in_tensor_size = in_tensor_size / (1024 ** 2)  # MB
        return in_tensor_size

    def get_model_size(self, model):
        torch.save(model, 'tmp_model_debug_size.pth')
        model_size = os.path.getsize('model_debug_size.pth')
        model_size = model_size / (1024 ** 2)  # MB
        return model_size


class PostProcessor(ABC):
    #
    def __init__(self, device, stride:list=None, anchor_grid=None, half=True, nkpt=5):
        self.stride = stride if stride else [8, 16, 32]
        self.stride = torch.tensor(self.stride).float().to(device)

        self.anchor_grid = anchor_grid if anchor_grid else torch.tensor(
            [[[[[[4., 5.]]],
               [[[6., 8.]]],
               [[[10., 12.]]]]],
             [[[[[15., 19.]]],
               [[[23., 30.]]],
               [[[39., 52.]]]]],
             [[[[[72., 97.]]],
               [[[123., 164.]]],
               [[[209., 297.]]]]]]).float().to(device)
        if half:
            self.stride = self.stride.half()
            self.anchor_grid = self.anchor_grid.half()
        self.nl = self.anchor_grid.shape[0]
        self.na = 3
        self.no = 21
        self.grid = [torch.zeros(1)] * 3
        self.nkpt = nkpt

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid(torch.arange(ny), torch.arange(nx), indexing='ij')
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def __call__(self, x):
        z = []
        for i in range(self.nl):
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            x_det = x[i][..., :6].clone()
            x_kpt = x[i][..., 6:].clone()

            if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
            kpt_grid_x = self.grid[i][..., 0:1]
            kpt_grid_y = self.grid[i][..., 1:2]

            if self.nkpt == 0:
                y = x[i].sigmoid()
            else:
                y = x_det.sigmoid()

            xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
            wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.na, 1, 1, 2)  # wh
            if self.nkpt != 0:
                x_kpt[..., 0::3] = (x_kpt[..., ::3] * 2. - 0.5 + kpt_grid_x.repeat(1, 1, 1, 1, self.nkpt)) * \
                                   self.stride[i]  # xy
                x_kpt[..., 1::3] = (x_kpt[..., 1::3] * 2. - 0.5 + kpt_grid_y.repeat(1, 1, 1, 1, self.nkpt)) * \
                                   self.stride[i]  # xy
                x_kpt[..., 2::3] = x_kpt[..., 2::3].sigmoid()

            y = torch.cat((xy, wh, y[..., 4:], x_kpt), dim=-1)

            z.append(y.view(bs, -1, self.no))

        return torch.cat(z, 1), x