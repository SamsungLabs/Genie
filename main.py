import logging
from models import get_model
from distill import distill_data
from utils import get_dataset, evaluate_classifier
import fire
import torch
from torch import nn
from reconstruct import quantize_model, reconstruct

logging.basicConfig(style='{', format='{asctime} {levelname:8} {name:20} {message}', datefmt='%H:%M:%S', level=logging.INFO)
log = logging.getLogger(__name__)

def main(
    train_path=None,
    val_path=None,
    model_name='resnet18',
    samples=1024, distill_batch=128, distill_iter=4000, lr_g=0.1, lr_z=0.01,
    bit_w=4, bit_a=4,
    recon_iter=20000, recon_batch=32, round_weight=1.0
):
    """Quantize the model with synthetic dataset.

    Args:
        train_path: Training set path. (If set, use training data rather than distilled images.)
        val_path: Validation set path.
        model_name: model architecture
        samples: # of distilled images to generate (or # of sampled training set)
        distill_batch: batch size at distillation
        distill_iter: # of iterations per distillation batch.
        lr_g: lr of generator
        lr_z: lr of latent vector
        bit_w: weight bitwidth
        bit_a: activation bitwidth
        recon_iter: # of iterations per reconstruction
        recon_batch: batch size at reconstruction
        round_weight: rounding loss weight for quantization
    """
    available_models = ('resnet18', 'resnet50', 'mobilenetv2', 'regnetx_600m', 'regnetx_3200m', 'mnasnet2.0', 'mnasnet1.0', 'mobilenetb')
    assert model_name in available_models, f'{model_name} not exist!'
    model = get_model(model_name, pretrained=True).cuda().eval()
    
    if train_path:
        train_set = get_dataset(train_path, samples)
        train_set = next(iter(torch.utils.data.DataLoader(
            train_set, batch_size=len(train_set), num_workers=4)))[0].cuda()
    else:
        train_set = distill_data(
            model, batch_size=distill_batch, total_samples=samples, lr_g=lr_g, lr_z=lr_z, iters=distill_iter)

    qmodel = quantize_model(
    	model, bit_w, bit_a,
    	# specify layer to quantize (nn.Identity: residual)
    	(nn.Conv2d, nn.Linear, nn.Identity)
    )

    reconstruct(
        model, qmodel, train_set,
    	# specify layer to reconstruct
        (
            'BasicBlock','Bottleneck', # resnet block
            'ResBottleneckBlock', # regnet block
            'DwsConvBlock', 'ConvBlock', # mobilenetb block
            'InvertedResidual', # mobilenetv2, mnasnet block
            'Linear', 'Conv2d' # default
        ), 
        round_weight=round_weight, iterations=recon_iter, batch_size=recon_batch
    )

    val_set = get_dataset(val_path)
    evaluate_classifier(val_set, qmodel)

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    fire.Fire(main)