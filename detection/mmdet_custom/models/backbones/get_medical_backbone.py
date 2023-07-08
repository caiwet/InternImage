import os
import sys
import torch
# sys.path.append(os.path.join(os.getcwd(), 'cxrlearn'))
from .cxrlearn import cxrlearn as cxrlearn

def get_gloria(finetune=False):
    """
    Backbone: GLORIA
    Model returns a tensor of size (B, 6), where B is the batch size and
    6 includes (x_c, y_c, x_et, y_et, distance between et and c, confidence of et present)
    """
    model = cxrlearn.gloria(
        model="resnet50",
        freeze_backbone=(finetune==False),
        num_ftrs=2048,
        num_out=None,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        )
    return model


def unit_test():
    # load model
    os.chdir(os.path.join(os.getcwd(), "cxrlearn"))
    model = get_gloria(finetune=True)
    os.chdir(os.path.join(os.getcwd(), ".."))

    # load image
    dataset = CXRDataset(
            root='/home/ec2-user/data/MIMIC-1105-512',
            image_dir='PNGImages',
            ann_file='annotations-512.json',
            transforms=get_transform(train=False),
            )
    img = dataset[100][0].unsqueeze_(0).to('cuda')
    print(model(img).shape)


if __name__ == '__main__':
    unit_test()
    # create random tensor of size (1, 2048)
    # x = torch.rand(1, 2048)