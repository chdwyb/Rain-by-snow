import os
import sys
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from config import Options

opt = Options()
path_result = opt.Result_Path_Test
path_target = opt.Target_Path_Test
image_list = os.listdir(path_target)
L = len(image_list)
total_psnr, total_ssim = 0, 0

for i in range(L):
    image_in = cv2.imread(path_result+str(image_list[i]), 1)
    image_tar = cv2.imread(path_target+str(image_list[i]), 1)

    psnr = peak_signal_noise_ratio(image_in, image_tar)
    ssim = structural_similarity(image_in/255., image_tar/255., channel_axis=-1)
    # ss = structural_similarity(image_in/255., image_tar/255., multichannel=True)

    total_psnr += psnr
    total_ssim += ssim

    sys.stdout.write(f'\r{(i+1)} / {L}, PSNR: {psnr}, SSIM: {ssim}\t')
    sys.stdout.flush()

print(f'\nPSNR: {total_psnr/L}, SSIM: {total_ssim/L}')