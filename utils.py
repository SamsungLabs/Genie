import torch
from torchvision import datasets, transforms

def find_parent(module, name: str):
    """Recursively apply getattr and returns parent of module"""
    if name == '':
        raise ValueError('Cannot Found')
    for sub_name in name.split('.')[: -1]:
        if hasattr(module, sub_name):
            module = getattr(module, sub_name)
        else:
            raise ValueError('submodule name not exist')
    return module

class ActivationHook():
    """
    Forward_hook used to get the output of the intermediate layer. 
    """

    def __init__(self, module):
        super(ActivationHook, self).__init__()
        self.inputs, self.outputs = None, None
        self.handle = module.register_forward_hook(self.hook)

    def hook(self, module, input, output):
        self.inputs = input[0]  # arg tuple
        self.outputs = output

    def remove(self):
        self.handle.remove()

def get_dataset(data_path, num_samples=None):
    data_transform = transforms.Compose([
        transforms.Resize(256, transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(data_path, data_transform)

    if num_samples is not None:
        # sample random subset from train dataset
        subset_indexes = torch.randperm(len(dataset))[:num_samples]
        dataset = torch.utils.data.Subset(dataset, subset_indexes)

    return dataset

@torch.no_grad()
def evaluate_classifier(dataset, model, batch_size=64, workers=4, print_freq=50):
    device = next(model.parameters()).device
    model.to(device).eval()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    total, correct = 0, 0
    for i, (images, target) in enumerate(data_loader):
        images = images.to(device)
        target = target.to(device)

        pred = model(images)
        correct += int((pred.argmax(dim=1)==target).sum())
        total += images.shape[0]

        if i % print_freq == 0:
            print(f"Test {i}/{len(data_loader)}: {correct/total*100:.2f}")

    print(f"Test: {correct/total*100:.2f}")
    return correct/total
