import numpy as np
import argparse
import os
from tqdm import tqdm
from load_raw import load_image
from metrics import ref_based, non_ref_based

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', type=str, required=True, help='Path for the Model Outputs')
    parser.add_argument('--reference-path', type=str, required=True, help='Path for Reference Images')
    args = parser.parse_args()

    dirs = os.listdir(args.output_path)

    print("Detected {} files in directory {}\n".format(len(dirs), args.output_path))
    file_names = []
    overall_psnr = []
    overall_psnr_l = []
    overall_ssim = []
    overall_uiqm = []
    overall_uciqe = []

    for file in tqdm(dirs):
        f = os.path.join(args.output_path, file)
        if f[-3:].lower() == "png":
            image = load_image(f)
            reference = load_image(os.path.join(args.reference_path, file[4:])) # outputs are of form out_X
            file_names.append(file[4:-4])

            uiqm,uciqe = non_ref_based(image)
            overall_uiqm.append(uiqm)
            overall_uciqe.append(uciqe)

            psnr, psnr_l, ssim = ref_based(image, reference)
            overall_psnr.append(psnr)
            overall_psnr_l.append(psnr_l)
            overall_ssim.append(ssim)

    print ("PSNR   : Mean: {:014.12f} std: {:.12f}".format(np.mean(overall_psnr), np.std(overall_psnr)))
    print ("PSNR_L : Mean: {:014.12f} std: {:.12f}".format(np.mean(overall_psnr_l), np.std(overall_psnr_l)))
    print ("SSIM   : Mean: {:014.12f} std: {:.12f}".format(np.mean(overall_ssim), np.std(overall_ssim)))
    print ("UIQM   : Mean: {:014.12f} std: {:.12f}".format(np.mean(overall_uiqm), np.std(overall_uiqm)))
    print ("UCIQE  : Mean: {:014.12f} std: {:.12f}".format(np.mean(overall_uciqe), np.std(overall_uciqe)))
