"""
    MobileNet for ImageNet-1K, implemented in PyTorch.
    Original paper: 'MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications,'
    https://arxiv.org/abs/1704.04861.
"""

__all__ = ['MobileNet', 'mobilenet_w1', 'get_mobilenet']

import hashlib
import logging
import os
import zipfile
from inspect import isfunction

import torch.nn as nn


_model_sha1 = {'mobilenetb_w1': ('0816', '107275a1173b201634cca077dd126a550bc99dae', 'v0.0.489', 'MobileNet(B) x1.0', '1704.04861', 'in1k', 224, 0.875, 200, '')}


def get_activation_layer(activation):
    """
    Create activation layer from string/function.
    Parameters:
    ----------
    activation : function, or str, or nn.Module
        Activation function or name of activation function.
    Returns:
    -------
    nn.Module
        Activation layer.
    """
    assert activation is not None
    if isfunction(activation):
        return activation()
    elif isinstance(activation, str):
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "relu6":
            return nn.ReLU6(inplace=True)
        elif activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "identity":
            return nn.Identity()
        else:
            raise NotImplementedError()
    else:
        assert isinstance(activation, nn.Module)
        return activation
    
    
class ConvBlock(nn.Module):
    """
    Standard convolution block with Batch normalization and activation.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int, or tuple/list of 2 int, or tuple/list of 4 int
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 groups=1,
                 bias=False,
                 use_bn=True,
                 bn_eps=1e-5,
                 activation=(lambda: nn.ReLU(inplace=True))):
        super(ConvBlock, self).__init__()
        self.activate = (activation is not None)
        self.use_bn = use_bn
        self.use_pad = (isinstance(padding, (list, tuple)) and (len(padding) == 4))

        if self.use_pad:
            self.pad = nn.ZeroPad2d(padding=padding)
            padding = 0
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
        if self.use_bn:
            self.bn = nn.BatchNorm2d(num_features=out_channels, eps=bn_eps)
        if self.activate:
            self.activ = get_activation_layer(activation)

    def forward(self, x):
        x = self.conv(self.pad(x) if self.use_pad else x)
        if self.use_bn:
            x = self.bn(x)
        return self.activ(x) if self.activate else x
    

def conv1x1_block(in_channels,
                  out_channels,
                  stride=1,
                  padding=0,
                  groups=1,
                  bias=False,
                  use_bn=True,
                  bn_eps=1e-5,
                  activation=(lambda: nn.ReLU(inplace=True))):
    """
    1x1 version of the standard convolution block.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int, or tuple/list of 2 int, or tuple/list of 4 int, default 0
        Padding value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation)
    

def conv3x3_block(in_channels,
                  out_channels,
                  stride=1,
                  padding=1,
                  dilation=1,
                  groups=1,
                  bias=False,
                  use_bn=True,
                  bn_eps=1e-5,
                  activation=(lambda: nn.ReLU(inplace=True))):
    """
    3x3 version of the standard convolution block.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int, or tuple/list of 2 int, or tuple/list of 4 int, default 1
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation)
    

def dwconv_block(in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=False,
                 use_bn=True,
                 bn_eps=1e-5,
                 activation=(lambda: nn.ReLU(inplace=True))):
    """
    Depthwise version of the standard convolution block.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int, or tuple/list of 2 int, or tuple/list of 4 int, default 1
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=out_channels,
        bias=bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation)
    

class DwsConvBlock(nn.Module):
    """
    Depthwise separable convolution block with BatchNorms and activations at each convolution layers.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int, or tuple/list of 2 int, or tuple/list of 4 int
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    bias : bool, default False
        Whether the layer uses a bias vector.
    dw_use_bn : bool, default True
        Whether to use BatchNorm layer (depthwise convolution block).
    pw_use_bn : bool, default True
        Whether to use BatchNorm layer (pointwise convolution block).
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    dw_activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function after the depthwise convolution block.
    pw_activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function after the pointwise convolution block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 bias=False,
                 dw_use_bn=True,
                 pw_use_bn=True,
                 bn_eps=1e-5,
                 dw_activation=(lambda: nn.ReLU(inplace=True)),
                 pw_activation=(lambda: nn.ReLU(inplace=True))):
        super(DwsConvBlock, self).__init__()
        self.dw_conv = dwconv_block(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            use_bn=dw_use_bn,
            bn_eps=bn_eps,
            activation=dw_activation)
        self.pw_conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=bias,
            use_bn=pw_use_bn,
            bn_eps=bn_eps,
            activation=pw_activation)

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x
    
    
def dwsconv3x3_block(in_channels,
                     out_channels,
                     stride=1,
                     padding=1,
                     dilation=1,
                     bias=False,
                     bn_eps=1e-5,
                     dw_activation=(lambda: nn.ReLU(inplace=True)),
                     pw_activation=(lambda: nn.ReLU(inplace=True)),
                     **kwargs):
    """
    3x3 depthwise separable version of the standard convolution block.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int, or tuple/list of 2 int, or tuple/list of 4 int, default 1
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    bias : bool, default False
        Whether the layer uses a bias vector.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    dw_activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function after the depthwise convolution block.
    pw_activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function after the pointwise convolution block.
    """
    return DwsConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        bn_eps=bn_eps,
        dw_activation=dw_activation,
        pw_activation=pw_activation,
        **kwargs)
    

class MobileNet(nn.Module):
    """
    MobileNet model from 'MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications,'
    https://arxiv.org/abs/1704.04861.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    first_stage_stride : bool
        Whether stride is used at the first stage.
    dw_use_bn : bool, default True
        Whether to use BatchNorm layer (depthwise convolution block).
    dw_activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function after the depthwise convolution block.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 first_stage_stride,
                 dw_use_bn=True,
                 dw_activation=(lambda: nn.ReLU(inplace=True)),
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(MobileNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        init_block_channels = channels[0][0]
        self.features.add_module("init_block", conv3x3_block(in_channels=in_channels, out_channels=init_block_channels, stride=2))
        in_channels = init_block_channels
        
        for i, channels_per_stage in enumerate(channels[1:]):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and ((i != 0) or first_stage_stride) else 1
                stage.add_module(f"unit{j + 1}", dwsconv3x3_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    dw_use_bn=dw_use_bn,
                    dw_activation=dw_activation))
                in_channels = out_channels
            self.features.add_module(f"stage{i + 1}", stage)
        self.features.add_module("final_pool", nn.AvgPool2d(kernel_size=7, stride=1))
        self.output = nn.Linear(in_features=in_channels, out_features=num_classes)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if 'dw_conv.conv' in name:
                nn.init.kaiming_normal_(module.weight, mode='fan_in')
            elif name == 'init_block.conv' or 'pw_conv.conv' in name:
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
            elif 'bn' in name:
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif 'output' in name:
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


def _check_sha1(file_name, sha1_hash):
    sha1 = hashlib.sha1()
    with open(file_name, "rb") as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    return sha1.hexdigest() == sha1_hash


def _download(url, path=None, overwrite=False, sha1_hash=None, retries=5, verify_ssl=True):
    """
    Download an given URL
    Parameters:
    ----------
    url : str
        URL to download
    path : str, optional
        Destination path to store downloaded file. By default stores to the
        current directory with same name as in url.
    overwrite : bool, optional
        Whether to overwrite destination file if already exists.
    sha1_hash : str, optional
        Expected sha1 hash in hexadecimal digits. Will ignore existing file when hash is specified
        but doesn't match.
    retries : integer, default 5
        The number of times to attempt the download in case of failure or non 200 return codes
    verify_ssl : bool, default True
        Verify SSL certificates.
    Returns:
    -------
    str
        The file path of the downloaded file.
    """
    import warnings
    try:
        import requests
    except ImportError:
        class requests_failed_to_import(object):
            pass
        requests = requests_failed_to_import

    if path is None:
        fname = url.split("/")[-1]
        # Empty filenames are invalid
        assert fname, "Can't construct file-name from this URL. Please set the `path` option manually."
    else:
        path = os.path.expanduser(path)
        fname = os.path.join(path, url.split('/')[-1]) if os.path.isdir(path) else path
    assert retries >= 0, "Number of retries should be at least 0"

    if not verify_ssl:
        warnings.warn("Unverified HTTPS request is being made (verify_ssl=False). Adding certificate verification is strongly advised.")

    if overwrite or not os.path.exists(fname) or (sha1_hash and not _check_sha1(fname, sha1_hash)):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        while retries + 1 > 0:
            # Disable pyling too broad Exception
            # pylint: disable=W0703
            try:
                print(f"Downloading {fname} from {url}...")
                r = requests.get(url, stream=True, verify=verify_ssl)
                
                if r.status_code != 200:
                    raise RuntimeError(f"Failed downloading url {url}")
                
                with open(fname, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                            
                if sha1_hash and not _check_sha1(fname, sha1_hash):
                    raise UserWarning(f"File {fname} is downloaded but the content hash does not match."
                                      " The repo may be outdated or download may be incomplete. "
                                      "If the `repo_url` is overridden, consider switching to "
                                      "the default repo.")
                break
            except Exception as e:
                retries -= 1
                if retries <= 0:
                    raise e
                else:
                    print(f"download failed, retrying, {retries} attempt{'s' if retries > 1 else ''} left")

    return fname


def get_model_name_suffix_data(model_name):
    if model_name not in _model_sha1:
        raise ValueError(f"Pretrained model for {model_name} is not available.")
    error, sha1_hash, repo_release_tag, _, _, _, _, _, _, _ = _model_sha1[model_name]
    return error, sha1_hash, repo_release_tag


def get_model_file(model_name, local_model_store_dir_path=os.path.join("~", ".torch", "models")):
    """
    Return location for the pretrained on local file system. This function will download from online model zoo when
    model cannot be found or has mismatch. The root directory will be created if it doesn't exist.
    Parameters:
    ----------
    model_name : str
        Name of the model.
    local_model_store_dir_path : str, default $TORCH_HOME/models
        Location for keeping the model parameters.
    Returns:
    -------
    file_path
        Path to the requested pretrained model file.
    """
    error, sha1_hash, repo_release_tag = get_model_name_suffix_data(model_name)
    short_sha1 = sha1_hash[:8]
    file_name = f"{model_name}-{error}-{short_sha1}.pth"
    local_model_store_dir_path = os.path.expanduser(local_model_store_dir_path)
    file_path = os.path.join(local_model_store_dir_path, file_name)
    if os.path.exists(file_path):
        if _check_sha1(file_path, sha1_hash):
            return file_path
        else:
            logging.warning("Mismatch in the content of model file detected. Downloading again.")
    else:
        logging.info(f"Model file not found. Downloading to {file_path}.")

    if not os.path.exists(local_model_store_dir_path):
        os.makedirs(local_model_store_dir_path)

    zip_file_path = file_path + ".zip"
    _download(url=f"https://github.com/osmr/imgclsmob/releases/download/{repo_release_tag}/{file_name}.zip", path=zip_file_path, overwrite=True, verify_ssl=False)
    
    with zipfile.ZipFile(zip_file_path) as zf:
        zf.extractall(local_model_store_dir_path)
    os.remove(zip_file_path)

    if _check_sha1(file_path, sha1_hash):
        return file_path
    else:
        raise ValueError("Downloaded file has different hash. Please try again.")
    

def load_model(net, file_path, ignore_extra=True):
    """
    Load model state dictionary from a file.
    Parameters:
    ----------
    net : Module
        Network in which weights are loaded.
    file_path : str
        Path to the file.
    ignore_extra : bool, default True
        Whether to silently ignore parameters from the file that are not present in this Module.
    """
    import torch

    if ignore_extra:
        pretrained_state = torch.load(file_path)
        model_dict = net.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items() if k in model_dict}
        net.load_state_dict(pretrained_state)
    else:
        net.load_state_dict(torch.load(file_path))


def download_model(net, model_name, local_model_store_dir_path=os.path.join("~", ".torch", "models"), ignore_extra=True):
    """
    Load model state dictionary from a file with downloading it if necessary.
    Parameters:
    ----------
    net : Module
        Network in which weights are loaded.
    model_name : str
        Name of the model.
    local_model_store_dir_path : str, default $TORCH_HOME/models
        Location for keeping the model parameters.
    ignore_extra : bool, default True
        Whether to silently ignore parameters from the file that are not present in this Module.
    """
    load_model(net=net, file_path=get_model_file(model_name=model_name, local_model_store_dir_path=local_model_store_dir_path), ignore_extra=ignore_extra)
    

def get_mobilenet(width_scale, dws_simplified=False, model_name=None, pretrained=False, root=os.path.join("~", ".torch", "models"), **kwargs):
    """
    Create MobileNet model with specific parameters.

    Parameters:
    ----------
    width_scale : float
        Scale factor for width of layers.
    dws_simplified : bool, default False
        Whether to use simplified depthwise separable convolution block.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    channels = [[32], [64], [128, 128], [256, 256], [512, 512, 512, 512, 512, 512], [1024, 1024]]
    first_stage_stride = False

    if width_scale != 1.0:
        channels = [[int(cij * width_scale) for cij in ci] for ci in channels]
        
    dw_use_bn = not dws_simplified
    dw_activation = None if dws_simplified else lambda: nn.ReLU(inplace=True)

    net = MobileNet(channels=channels, first_stage_stride=first_stage_stride, dw_use_bn=dw_use_bn, dw_activation=dw_activation, **kwargs)

    if pretrained:
        if model_name is None or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        download_model(net=net, model_name=model_name, local_model_store_dir_path=root)

    return net

def mobilenetb_w1(**kwargs):
    """
    1.0 MobileNet(B)-224 model with simplified depthwise separable convolution block from 'MobileNets: Efficient
    Convolutional Neural Networks for Mobile Vision Applications,' https://arxiv.org/abs/1704.04861.
    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_mobilenet(width_scale=1.0, dws_simplified=True, model_name="mobilenetb_w1", **kwargs)


