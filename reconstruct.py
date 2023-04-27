import logging
import copy
import torch
from torch import nn, Tensor
from collections import OrderedDict
from utils import ActivationHook, find_parent
from quantizer import WeightQuantizer, ActivationQuantizer

log = logging.getLogger(__name__)

class QuantizableLayer(nn.Module):
    """
    Wrapper module that performs fake quantization operations.
    """

    def __init__(self, org_module, weight_bits, activation_bits):
        super(QuantizableLayer, self).__init__()

        self.org_module = org_module
        self.enable_act_quant = False
        self.weight_quantizer = None
        self.activation_quantizer = None

        if weight_bits is not None:
            if hasattr(self.org_module, 'weight'):
                self.weight_quantizer = WeightQuantizer(self.org_module.weight, weight_bits)
                self.org_module._parameters.pop('weight', None)

        if activation_bits is not None:
            self.activation_quantizer = ActivationQuantizer(activation_bits)

    def forward(self, x: Tensor):
        if self.activation_quantizer and self.enable_act_quant:
            # trick to share same quantization parameter between residual and conv
            if hasattr(x, 'tensor_quantizer'):
                assert self.activation_quantizer.n_bits <= x.tensor_quantizer.n_bits, 'Activation bitwidth becomes smaller.'
                self.activation_quantizer = x.tensor_quantizer
            else:
                x.tensor_quantizer = self.activation_quantizer
            x = self.activation_quantizer(x)

        if self.weight_quantizer:
            self.org_module.weight = self.weight_quantizer()
        x = self.org_module(x)

        return x

class LinearTempDecay:
    def __init__(self, iter_max, rel_start_decay, start_t, end_t):
        self.t_max = iter_max
        self.start_decay = rel_start_decay * iter_max
        self.start_b = start_t
        self.end_b = end_t

    def __call__(self, cur_iter):
        if cur_iter < self.start_decay:
            return self.start_b
        else:
            rel_t = (cur_iter-self.start_decay) / (self.t_max-self.start_decay)
            return self.end_b + (self.start_b-self.end_b)*max(0.0, 1 - rel_t)

def reconstruct_block(
    block, x, x_q, y, 
    iterations=20000, round_weight=1.0,
    lr_w_scale=0.0001, lr_a_scale=0.00004, lr_bit=0.001,
    annealing_range=(20,2), annealing_warmup=0.2, batch_size=32
):
    param_a_scale = []
    param_w_scale = []
    param_bit = []

    for module in block.modules():
        if isinstance(module, WeightQuantizer):
            module.train_mode = True
            param_w_scale.append(module.scale)
            param_bit.append(module.bit_logit)

        elif isinstance(module, ActivationQuantizer):
            module.train_mode = True
            param_a_scale.append(module.scale)

    opt_scale = torch.optim.Adam([
        {"params": param_w_scale, 'lr': lr_w_scale},
        {"params": param_a_scale, 'lr': lr_a_scale}
    ])

    opt_bit = torch.optim.Adam(param_bit, lr=lr_bit)
    scheduler_scale = torch.optim.lr_scheduler.CosineAnnealingLR(opt_scale, T_max=iterations)

    temp_decay = LinearTempDecay(
        iterations, rel_start_decay=annealing_warmup,
        start_t=annealing_range[0], end_t=annealing_range[1])

    iters = 0
    while iters < iterations:
        perms = torch.randperm(len(x)).view(batch_size, -1)
        for idx in perms:
            iters += 1

            x_mix = torch.where(torch.rand_like(x[idx]) < 0.5, x_q[idx], x[idx]) # use QDrop
            y_q = block(x_mix)
            
            recon_loss = (y_q - y[idx]).pow(2).sum(1).mean()
            round_loss = 0

            annealing_temp = temp_decay(iters)
            if iters >= annealing_warmup*iterations:
                for module in block.modules():
                    if isinstance(module, WeightQuantizer):
                        round_loss += (1 - (2*module.soft_target() - 1).abs().pow(annealing_temp)).sum()

            total_loss = recon_loss + round_loss * round_weight

            opt_scale.zero_grad()
            opt_bit.zero_grad()
            total_loss.backward()
            opt_scale.step()
            opt_bit.step()
            scheduler_scale.step()

            if iters == 1 or iters % 1000 == 0:
                log.info(
                    f'{iters}/{iterations}, Total loss: {total_loss:.3f} (rec:{recon_loss:.3f}, round:{round_loss:.3f})'
                    +f'\tb={annealing_temp:.2f}\tcount={iters}')

            if iters >= iterations:
                break

    # Finish optimization, use hard rounding.
    for module in block.modules():
        if isinstance(module, (WeightQuantizer, ActivationQuantizer)):
            module.train_mode = False


def quantize_model(model, bit_w, bit_a, quant_ops):
    """Convert a full precision model to a quantized model.

    Args:
        model: full precision model
        bit_w: weight bitwidth
        bit_a: activation bitwidth
        quant_ops (tuple): List of operator types to quantize

    Returns:
        nn.Module: quantized model
    """
    qmodel = copy.deepcopy(model)
    qconfigs = []
    for name, module in qmodel.named_modules():
        if isinstance(module, quant_ops):
            qconfigs.append({'name': name, 'module': module, 'bit_w': bit_w, 'bit_a': bit_a})
            parent = find_parent(qmodel, name)

    # keep first and last layer to 8bit
    qconfigs[0] = {**qconfigs[0], 'bit_w': 8, 'bit_a': None} # Input image is already quantized.
    # if you want to use QDrop quantization setting, comment out the code below
    qconfigs[1] = {**qconfigs[1], 'bit_a': 8} # BRECQ keeps the second layer’s input to 8bit
    qconfigs[-1] = {**qconfigs[-1], 'bit_w': 8, 'bit_a': 8}

    for qconfig in qconfigs:
        parent = find_parent(qmodel, qconfig['name'])
        setattr(parent, qconfig['name'].split('.')[-1], 
                QuantizableLayer(qconfig['module'], qconfig['bit_w'], qconfig['bit_a']))

    return qmodel

def reconstruct(teacher, student, cali_data, reconstruct_unit, **kwargs):
    """Reconstructs the quantized model.

    Args:
        teacher: full precision model
        student: quantized model to reconstruct
        cali_data (tensor): calibration dataset
        reconstruct_unit (tuple): A list of block or layer to reconstruct
    """
    teacher_modules = OrderedDict(teacher.named_modules())
    student_modules = OrderedDict(student.named_modules())
    reconstruct_pair = []

    visited = set()
    for name, module in teacher_modules.items():
        if (module in reconstruct_unit or module.__class__.__name__ in reconstruct_unit) and module not in visited:
            visited.update(module.modules())
            reconstruct_pair.append((module, student_modules[name], name))

    for i, (teacher_block, student_block, name) in enumerate(reconstruct_pair):
        log.info(f'Recontruct ({i}/{len(reconstruct_pair)}): {name}')
        for name, module in student_block.named_modules():
            if isinstance(module, QuantizableLayer):
                module.enable_act_quant = True
            elif isinstance(module, (WeightQuantizer, ActivationQuantizer)):
                module.train_mode = True

        act_x, act_y, act_x_q = [], [], []
        batch_size = 32
        cali_data_slices = cali_data.view(*(-1, batch_size, *cali_data.shape[1:]))

        t_hook = ActivationHook(teacher_block)
        s_hook = ActivationHook(student_block)

        with torch.no_grad():
            for x in cali_data_slices:
                teacher(x)
                student(x)
                act_x.append(t_hook.inputs)
                act_y.append(t_hook.outputs)
                act_x_q.append(s_hook.inputs)
        act_x = torch.cat(act_x)
        act_y = torch.cat(act_y)
        act_x_q = torch.cat(act_x_q)
                
        t_hook.remove()
        s_hook.remove()
        reconstruct_block(student_block, act_x, act_x_q, act_y, **kwargs)

    for name, module in student.named_modules():
        if isinstance(module, QuantizableLayer):
            module.enable_act_quant = True
        elif isinstance(module, (WeightQuantizer, ActivationQuantizer)):
            module.train_mode = False
