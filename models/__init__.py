import torch
import models.mnasnet, models.mobilenetv2, models.regnet, models.resnet, models.mobilenetb

archs = {
    'resnet18': models.resnet.resnet18,
    'resnet50': models.resnet.resnet50,
    'mobilenetv2': models.mobilenetv2.mobilenetv2,
    'regnetx_600m': models.regnet.regnetx_600m,
    'regnetx_3200m': models.regnet.regnetx_3200m,
    'mnasnet2.0': models.mnasnet.mnasnet,
    'mnasnet1.0': models.mnasnet.mnasnet1_0,
    'mobilenetb': models.mobilenetb.mobilenetb_w1,
}

pretrained_urls = {
    'resnet18': 'https://github.com/yhhhli/BRECQ/releases/download/v1.0/resnet18_imagenet.pth.tar',
    'resnet50': 'https://github.com/yhhhli/BRECQ/releases/download/v1.0/resnet50_imagenet.pth.tar',
    'mobilenetv2': 'https://github.com/yhhhli/BRECQ/releases/download/v1.0/mobilenetv2.pth.tar',
    'regnetx_600m': 'https://github.com/yhhhli/BRECQ/releases/download/v1.0/regnet_600m.pth.tar',
    'regnetx_3200m': 'https://github.com/yhhhli/BRECQ/releases/download/v1.0/regnet_3200m.pth.tar',
    'mnasnet2.0': 'https://github.com/yhhhli/BRECQ/releases/download/v1.0/mnasnet.pth.tar',
    'mnasnet1.0': 'https://download.pytorch.org/models/mnasnet1.0_top1_73.512-f206786ef8.pth',
}

def get_model(model_name: str, pretrained=False, **kwargs):
    # Call the model, load pretrained weights
    model = archs[model_name](**kwargs)
    if pretrained:
        if model_name == 'mobilenetb':
            return archs[model_name](pretrained=True)
        
        checkpoint = torch.hub.load_state_dict_from_url(url=pretrained_urls[model_name], map_location='cpu', progress=True)
        if model_name == 'mobilenetv2':
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
    return model

