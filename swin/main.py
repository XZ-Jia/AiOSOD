import os
import torch
import Training
import Testing
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--Training', default=True, type=bool, help='Training or not')
    parser.add_argument('--mode', default='swin_tiny', type=str, help='select pretrained model')
    parser.add_argument('--works', default=4, type=int, help='works num')
    parser.add_argument('--batch_size', default=16, type=int, help='batch_size')
    parser.add_argument('--epochs', default=800, type=int, help='epochs')
    parser.add_argument('--gpu', default='0', type=str, help='')
    parser.add_argument('--lr', default=1e-4, type=int, help='learning rate')
    parser.add_argument('--lr_decay_gamma', default=0.1, type=int, help='learning rate decay')
    parser.add_argument('--train_steps', default=100000, type=int, help='train_steps')
    parser.add_argument('--init_method', default='tcp://127.0.0.1:1221', type=str, help='init_method')
    parser.add_argument('--data_root', default='../Data/', type=str, help='data path')
    parser.add_argument('--img_size', default=224, type=int, help='network input size')
    parser.add_argument('--pretrained_model', default='swin_tiny_patch4_window7_224.pth', type=str, help='load pretrained model')
    parser.add_argument('--val_set', type=str, default='NLPR')
    parser.add_argument('--stepvalue1', default=50000, type=int, help='the step 1 for adjusting lr')
    parser.add_argument('--stepvalue2', default=70000, type=int, help='the step 2 for adjusting lr')
    parser.add_argument('--trainset', default='NJUD+NLPR+DUTLF-Depth+DUTS-TR+VT5000', type=str, help='Trainging set')
    parser.add_argument('--save_model_dir', default='checkpoint/', type=str, help='save model path')
    parser.add_argument('--Testing', default=True, type=bool, help='Testing or not')
    parser.add_argument('--save_test_path_root', default='preds/', type=str, help='save saliency maps path')
    parser.add_argument('--checkpoint', default='swin_tiny_best.pth', type=str, help='save saliency maps path')
    parser.add_argument('--test_paths', type=str, default='NJUD+NLPR+DUTLF-Depth+RGBD135+SSD+LFSD+ReDWeb-S+SIP+ECSSD+SOD+PASCAL-S+DUTS-TE+DUT-OMRON+HKU-IS+VT821+VT1000+VT5000')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.Training:
        Training.train_net(num_gpus=1, args=args)
    if args.Testing:
        Testing.test_net(args)
