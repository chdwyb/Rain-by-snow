import os
import sys
import cv2
import argparse
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def calculate_psnr_ssim(args):

    path_result = args.result_dir
    path_target = args.target_dir
    image_list = os.listdir(path_target)
    L = len(image_list)
    total_psnr, total_ssim = 0, 0

    for i in range(L):
        image_input = cv2.imread(os.path.join(path_result, str(image_list[i])), 1)
        image_target = cv2.imread(os.path.join(path_target, str(image_list[i])), 1)

        psnr = peak_signal_noise_ratio(image_input, image_target)
        ssim = structural_similarity(image_input / 255., image_target / 255., channel_axis=-1)
        # for different skimage version
        # ssin = structural_similarity(image_in/255., image_tar/255., multichannel=True)

        total_psnr += psnr
        total_ssim += ssim

        sys.stdout.write(f'\r{(i+1)} / {L}, PSNR: {psnr}, SSIM: {ssim}\t')
        sys.stdout.flush()

    print(f'\nPSNR: {total_psnr / L}, SSIM: {total_ssim / L}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, default='./RainCityscapes/test/result')
    parser.add_argument('--target_dir', type=str, default='./RainCityscapes/test/target')
    args = parser.parse_args()

    calculate_psnr_ssim(args)


