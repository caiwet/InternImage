import torch
import requests
def get_model_from_link(model_url, save_to):
    response = requests.get(model_url)

    if response.status_code == 200:
        with open(save_to, 'wb') as file:
            file.write(response.content)
    else:
        print(f"Failed to download the model from {model_url}")

def model():
    cls_model = torch.load('/home/cat302/ETT-Project/InternImage/classification/temp_train/internimage_t_1k_224/ckpt_epoch_best.pth')
    pretrained_model = torch.load('pretrained.pth')
    for key in pretrained_model['model'].keys():
        if key in cls_model['model'].keys():
            expected_shape = pretrained_model['model'][key].size()
            actual_shape = cls_model['model'][key].size()
            if 'head' in key:
                continue
            elif actual_shape != expected_shape:
                print(expected_shape)
                print(actual_shape)
            else:
                print(key)
                pretrained_model['model'][key] = cls_model['model'][key]
    
    torch.save(pretrained_model, 'modified_pretrained.pth')


if __name__ == "__main__":
    # pretrained = 'https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_t_1k_224.pth'
    # get_model_from_link(pretrained, save_to='pretrained.pth')
    model()