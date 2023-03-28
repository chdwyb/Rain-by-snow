import sys
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from RSFormer import RSFormer
from datasets import *
from config import Options
from utils import pad, unpad


if __name__ == '__main__':

    opt = Options()

    inputPathTest = opt.Input_Path_Test
    resultPathTest = opt.Result_Path_Test
    modelPath = opt.MODEL_PATH

    myNet = RSFormer()
    myNet = nn.DataParallel(myNet)
    if opt.CUDA_USE:
        myNet = myNet.cuda()

    datasetTest = MyTestDataSet(inputPathTest)
    testLoader = DataLoader(dataset=datasetTest, batch_size=1, shuffle=False, drop_last=False,
                            num_workers=opt.Num_Works, pin_memory=True)

    print('--------------------------------------------------------------')
    # pretrained model
    if opt.CUDA_USE:
        myNet.load_state_dict(torch.load(modelPath))
    else:
        myNet.load_state_dict(torch.load(modelPath, map_location=torch.device('cpu')))
    myNet.eval()

    with torch.no_grad():
        timeStart = time.time()
        for index, (x, name) in enumerate(tqdm(testLoader, desc='Testing !!! ', file=sys.stdout), 0):
            torch.cuda.empty_cache()

            input_test = x.cuda() if opt.CUDA_USE else x

            input_test, pad_size = pad(input_test, factor=16)
            output_test = myNet(input_test)
            output_test = unpad(output_test, pad_size)

            save_image(output_test, resultPathTest + name[0])
        timeEnd = time.time()
        print('---------------------------------------------------------')
        print("Testing Process Finished !!! Time: {:.4f} s".format(timeEnd - timeStart))
