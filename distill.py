import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torch
import copy
from utils import ActivationHook, find_parent

log = logging.getLogger(__name__)

class Generator(nn.Module):
    def __init__(self, image_size=224, latent_dim=256, c1=128, c2=64):
        super(Generator, self).__init__()

        self.init_size = image_size // 4
        self.linear = nn.Linear(latent_dim, c1 * self.init_size ** 2)
        self.conv_layers = nn.Sequential(
            nn.BatchNorm2d(c1),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(c1, c2, 3, stride=1, padding=1),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(c2, 3, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(3, affine=False)
        )

    def forward(self, z):
        out = self.linear(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        out = self.conv_layers(out)
        return out

class SwingConv2d(nn.Module):
    def __init__(self, org_module, jitter_size=1):
        super(SwingConv2d, self).__init__()
        self.org_module = org_module
        self.jitter_size = jitter_size

    def forward(self, x):
        src_x = np.random.randint(self.jitter_size*2+1)
        src_y = np.random.randint(self.jitter_size*2+1)
        input_pad = F.pad(x, [self.jitter_size for i in range(4)], mode='reflect')
        input_new = input_pad[:, :, src_y:src_y+x.shape[2], src_x:src_x+x.shape[3]] 
        assert input_new.shape == x.shape, f'{input_new.shape}, {input_pad.shape}, {x.shape}'
        return self.org_module(input_new)

def l2_loss(A, B):
    return (A - B).norm()**2 / B.size(0)

def distill_data(model, batch_size, total_samples, lr_g=0.1, lr_z=0.01, iters=4000):
    """Generate synthetic dataset using distillation

    Args:
        model: model to be distilled
        batch_size: batch size at distillation
        total_samples: # of images to generate
        lr_g: lr of generator
        lr_z: lr of latent vector
        iters: # of iterations per distillation batch.

    Returns:
        Tensor: synthetic dataset (dim: total_samples x 3 x 224 x 224)
    """
    latent_dim = 256
    eps = 1e-6

    model = copy.deepcopy(model).cuda().eval()
    
    hooks, bn_stats = [], []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if module.stride != (1, 1):
                parent = find_parent(model, name)
                setattr(parent, name.split('.')[-1], SwingConv2d(module, jitter_size=1))

        elif isinstance(module, nn.BatchNorm2d):
            hooks.append(ActivationHook(module))
            bn_stats.append((module.running_mean.detach().clone().cuda(),
                            torch.sqrt(module.running_var + eps).detach().clone().cuda()))

    dataset = []
    for i in range(total_samples//batch_size):
        log.info(f'Generate Image ({i*batch_size}/{total_samples})')
        # initialize the criterion, optimizer, and scheduler
        z = torch.randn(batch_size, latent_dim).cuda().requires_grad_()
        generator = Generator(latent_dim=latent_dim).cuda()

        opt_z = optim.Adam([z], lr=lr_g)
        scheduler_z = optim.lr_scheduler.ReduceLROnPlateau(opt_z, min_lr=1e-4, verbose=False, patience=100)
        opt_g = optim.Adam(generator.parameters(), lr=lr_z)
        scheduler_g = optim.lr_scheduler.ExponentialLR(opt_g, gamma=0.95)

        input_mean = torch.zeros(1, 3).cuda()
        input_std = torch.ones(1, 3).cuda()

        for it in range(iters):
            model.zero_grad()
            opt_z.zero_grad()
            opt_g.zero_grad()

            x = generator(z)
            model(x)

            mean_loss, std_loss = 0, 0
            data_std, data_mean = torch.std_mean(x, [2, 3])
            mean_loss += l2_loss(input_mean, data_mean)
            std_loss += l2_loss(input_std, data_std)

            for (bn_mean, bn_std), hook in zip(bn_stats, hooks):
                bn_input = hook.inputs
                data_std, data_mean = torch.std_mean(bn_input, [0, 2, 3])
                mean_loss += l2_loss(bn_mean, data_mean)
                std_loss += l2_loss(bn_std, data_std)

            total_loss = mean_loss + std_loss
            total_loss.backward()
            opt_z.step()
            opt_g.step()
            scheduler_z.step(total_loss.item())

            if (it+1) % 100 == 0:
                log.info(f'{it+1}/{iters}, Loss: {total_loss:.3f}, Mean: {mean_loss:.3f}, Std: {std_loss:.3f}')
                scheduler_g.step()

        dataset.append(x.detach().clone())

    for hook in hooks:
        hook.remove()

    dataset = torch.cat(dataset)
    return dataset
