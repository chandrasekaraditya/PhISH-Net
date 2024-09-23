import os, cv2, torch
import numpy as np
import skimage.exposure
from torchvision import transforms
from tqdm import tqdm
from model import HDRPointwiseNN
from utils import load_image, resize, load_params
import glob

def test(ckpt, args={}):
    state_dict = torch.load(ckpt)
    state_dict, params = load_params(state_dict)
    params.update(args)

    device = torch.device("cuda")
    tensor = transforms.Compose([
        transforms.ToTensor(),])

    if os.path.isdir(params['input']):
        test_files = glob.glob(params['input']+'/*')
    else:
        test_files = [params['input']]

    t_100 = 0
    for img_path in tqdm(test_files):
        img_name = img_path.split('/')[-1]

        low = tensor(resize(load_image(img_path),params['net_input_size'],strict=True).astype(np.float32)).repeat(1,1,1,1)/255
        full = tensor(load_image(img_path).astype(np.float32)).repeat(1,1,1,1)/255

        low = low.to(device)
        full = full.to(device)
        with torch.no_grad():
            model = HDRPointwiseNN(params=params)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            model.to(device)
            
            res, _ = model(low, full)
            res = torch.clip(res, min=full, max=torch.ones(full.shape).to(device))
            img =  torch.div(full, torch.add(res, 0.001))
            res = (res.cpu().detach().numpy()).transpose(0,2,3,1)[0]
            res = skimage.exposure.rescale_intensity(res, out_range=(0.0,255.0)).astype(np.uint8)
            res = (res*255.0).astype(np.uint8)

            img = (img.cpu().detach().numpy()).transpose(0,2,3,1)[0]
            img = skimage.exposure.rescale_intensity(img, out_range=(0.0,255.0)).astype(np.uint8)
            cv2.imwrite(os.path.join(params['output'],f'out_{img_name}'), img[...,::-1])

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='PhISH-Net Inference')
    parser.add_argument('--checkpoint', type=str, default='../checkpoints/PhISH-Net.pth',  help='model state path')
    parser.add_argument('--input', type=str, default='../data/input', help='image path')
    parser.add_argument('--output', type=str, default='../data/output' , help='output image path')

    args = vars(parser.parse_args())

    test(args['checkpoint'], args)
