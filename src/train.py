import os, cv2, torch
import random, glob
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
import skimage.exposure
from torchvision import transforms
from tqdm import tqdm

from dataset import HDRDataset
from metrics import psnr
from model import HDRPointwiseNN
from utils import load_image, save_params, get_latest_ckpt, load_params, tv_loss, cos_loss, resize, load_params

torch.manual_seed(13)
random.seed(13)
torch.autograd.set_detect_anomaly(True)

def test(ckpt, args={}):
    state_dict = torch.load(ckpt)
    state_dict, params = load_params(state_dict)
    params.update(args)

    device = torch.device("cuda")
    tensor = transforms.Compose([
        transforms.ToTensor(),])

    if os.path.isdir(params['test_image']):
        test_files = glob.glob(params['input']+'/*')
    else:
        test_files = [params['test_image']]


    for img_path in test_files:
        img_name = img_path.split('/')[-1]
        print(f'Testing image: {img_name}')
        low = tensor(resize(load_image(img_path),params['net_input_size'],strict=True).astype(np.float32)).repeat(1,1,1,1)/255
        full = tensor(load_image(img_path).astype(np.float32)).repeat(1,1,1,1)/255

        low = low.to(device)
        full = full.to(device)
        with torch.no_grad():
            model = HDRPointwiseNN(params=params)
            model.load_state_dict(state_dict)
            model.eval()
            model.to(device)
            res, _ = model(low, full)
            res = torch.clip(res, min=full, max=torch.ones(full.shape).to(device))
            img =  torch.div(full, torch.add(res, 0.001))
            res = (res.cpu().detach().numpy()).transpose(0,2,3,1)[0]
            res = skimage.exposure.rescale_intensity(res, out_range=(0.0,255.0)).astype(np.uint8)
            cv2.imwrite(os.path.join(params['test_out'],f'illum_{img_name}'), res[...,::-1])

            img = (img.cpu().detach().numpy()).transpose(0,2,3,1)[0]
            img = skimage.exposure.rescale_intensity(img, out_range=(0.0,255.0)).astype(np.uint8)
            cv2.imwrite(os.path.join(params['test_out'],f'out_{img_name}'), img[...,::-1])

def train(params=None):
    os.makedirs(params['ckpt_path'], exist_ok=True)

    device = torch.device("cuda")

    train_dataset = HDRDataset(params['dataset'], params=params, suffix=params['dataset_suffix'], split='train')
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

    model = HDRPointwiseNN(params=params)

    if params['resume']:
        ckpt = get_latest_ckpt(params['ckpt_path'])
        print('Loading previous state:', ckpt)
        state_dict = torch.load(ckpt)
        state_dict,_ = load_params(state_dict)
        model.load_state_dict(state_dict)
    model.to(device)

    mseloss = torch.nn.MSELoss()
    optimizer = Adam(model.parameters(), params['lr'], eps=1e-7)

    count = 0
    for e in tqdm(range(params['epochs'])):
        model.train()
        for i, (low, full, target, depth) in enumerate(train_loader):
            optimizer.zero_grad()

            low = low.to(device)
            full = full.to(device)
            t = target.to(device)
            depth = depth.to(device)

            illum, z_params = model(low, full)

            res = torch.clip(illum, min=full, max=torch.ones(full.shape).to(device))
            out =  torch.div(full, torch.add(res, 0.001))

            loss_l2 = mseloss(t, out)
            loss_cos = cos_loss(t, out)
            loss_tv = tv_loss(full, res, 1)

            total_loss = loss_l2 * 10  
            total_loss= total_loss + loss_cos * 1 
            total_loss = total_loss + loss_tv * 2

            depth = depth.unsqueeze(1)
            a, b, c, d = z_params[:,0], z_params[:,1], z_params[:,2], z_params[:,3]
            a, b, c, d = a.unsqueeze(1).unsqueeze(1).unsqueeze(1), b.unsqueeze(1).unsqueeze(1).unsqueeze(1), c.unsqueeze(1).unsqueeze(1).unsqueeze(1), d.unsqueeze(1).unsqueeze(1).unsqueeze(1)

            a, c = torch.nn.functional.relu(a), torch.nn.functional.relu(c)
            b, d = torch.nn.functional.relu(b), torch.nn.functional.relu(d) 

            beta_D = a * torch.exp(-b * depth) + c * torch.exp(-d * depth)
            illum_clipped = torch.clip(illum, min=torch.zeros(full.shape).to(device), max=torch.ones(full.shape).to(device))

            max_val = 1.0
            beta_D = torch.stack([beta_D, beta_D, beta_D], axis=1).squeeze(2)
            beta_D_coarse = torch.min(torch.ones_like(illum)*max_val, -torch.log(illum_clipped+1e-8)/ (torch.max(torch.zeros_like(depth), depth)+1e-8))

            loss_beta = mseloss(beta_D, beta_D_coarse)
            
            total_loss = total_loss + loss_beta * 0.5
            total_loss.backward()

            if (count+1) % params['log_interval'] == 0:
                if count+1 == params['log_interval']:
                  print("Epoch, Iterations, Loss, PSNR", "Individual losses")
                _psnr = psnr(res,t).item()
                loss = total_loss.item()
                print(e, count, loss, _psnr, loss_l2.item(), loss_cos.item())

            optimizer.step()
            if (count+1) % params['ckpt_interval'] == 0:
                print('@@ MIN:',torch.min(res),'MAX:',torch.max(res))
                model.eval().cpu()
                ckpt_model_filename = "ckpt_"+str(e)+'_' + str(count) + ".pth"
                ckpt_model_path = os.path.join(params['ckpt_path'], ckpt_model_filename)
                state = save_params(model.state_dict(), params)
                torch.save(state, ckpt_model_path)
                test(ckpt_model_path)
                model.to(device).train()
            count += 1

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='PhISHNet Training')
    parser.add_argument('--resume', type=bool, default=False, help='Continue training from latest checkpoint')
    parser.add_argument('--ckpt-path', type=str, default='./checkpoints', help='Model checkpoint path')
    parser.add_argument('--test-image', type=str,default='data/input/99_img_.png', dest="test_image", help='Test image path')
    parser.add_argument('--test-out', type=str, default='data/output', dest="test_out", help='Output test image path')

    parser.add_argument('--luma-bins', type=int, default=8)
    parser.add_argument('--channel-multiplier', default=1, type=int)
    parser.add_argument('--spatial-bin', type=int, default=16)
    parser.add_argument('--guide-complexity', type=int, default=16)
    parser.add_argument('--batch-norm', action='store_true', help='If set use batch norm')
    parser.add_argument('--net-input-size', type=int, default=256, help='Size of low-res input')
    parser.add_argument('--net-output-size', type=int, default=512, help='Size of full-res input/output')

    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--ckpt-interval', type=int, default=100)
    parser.add_argument('--dataset', type=str, default='data', help='Dataset path with input/output dirs')
    parser.add_argument('--dataset-suffix', type=str, default='', help='Add suffix to input/output dirs. Useful when train on different dataset image sizes')

    params = vars(parser.parse_args())

    print('PARAMS:')
    print(params)

    train(params=params)
