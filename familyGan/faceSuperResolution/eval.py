import torch
from torch import nn
import argparse
from familyGan.faceSuperResolution.dataloader import CelebDataSet
from torch.utils.data import DataLoader
from familyGan.faceSuperResolution.model import Generator
import os
import sys
from torchvision import utils
from math import log10
from familyGan.faceSuperResolution.ssim import ssim, msssim

def test(dataloader, generator, MSE_Loss, step, alpha):
    avg_psnr = 0
    avg_ssim = 0
    avg_msssim = 0
    for i, (x2_target_image, x4_target_image, target_image, input_image) in enumerate(dataloader):
        if step==1:
            target_image = x2_target_image.to(device)
        elif step==2:
            target_image = x4_target_image.to(device)
        else:
            target_image = target_image.to(device)

        input_image = input_image.to(device)
        predicted_image = generator(input_image, step, alpha)
        mse_loss = MSE_Loss(0.5*predicted_image+0.5, 0.5*target_image+0.5)
        psnr = 10*log10(1./mse_loss.item())
        avg_psnr += psnr
        _ssim = ssim(0.5*predicted_image+0.5, 0.5*target_image+0.5)
        avg_ssim += _ssim.item()
        ms_ssim = msssim(0.5*predicted_image+0.5, 0.5*target_image+0.5)
        avg_msssim += ms_ssim.item()

        sys.stdout.write('\r [%d/%d] Test progress... PSNR: %6.4f'%(i, len(dataloader), psnr))
        utils.save_image(0.5*predicted_image+0.5, os.path.join(args.result_path, '%d_results.jpg'%i))
    print('Test done, Average PSNR:%6.4f, Average SSIM:%6.4f, Average MS-SSIM:%6.4f '%(avg_psnr/len(dataloader),avg_ssim/len(dataloader), avg_msssim/len(dataloader)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Implemnetation of Progressive Face Super-Resolution Attention to Face Landmarks')
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--checkpoint-path', default='./checkpoints/', type=str)
    parser.add_argument('--data-path', default='./dataset/', type=str)
    parser.add_argument('--result-path', default='./result/', type=str)
    parser.add_argument('--workers', default=4, type=int)
    args = parser.parse_args()

    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)
        print('===>make directory', args.result_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = CelebDataSet(data_path=args.data_path, state='test')
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    generator = Generator().to(device)
    g_checkpoint = torch.load(args.checkpoint_path)
    generator.load_state_dict(g_checkpoint['model_state_dict'], strict=False)
    step = g_checkpoint['step']
    alpha = g_checkpoint['alpha']
    iteration = g_checkpoint['iteration']

    print('pre-trained model is loaded step:%d, iteration:%d'%(step, iteration))
    MSE_Loss = nn.MSELoss()

    test(dataloader, generator, MSE_Loss, step, alpha)
