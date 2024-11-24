import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.multiprocessing as mp
import torch.distributed as dist
from datetime import datetime
from dataset import get_loader
import math
from Models.ImageDepthNet import ImageDepthNet
import os
import numpy as np
from torch.utils import data
import logging
def save_loss(save_dir, whole_iter_num, epoch_total_loss, epoch_loss, epoch):
    fh = open(save_dir, 'a')
    epoch_total_loss = str(epoch_total_loss)
    epoch_loss = str(epoch_loss)
    fh.write('until_' + str(epoch) + '_run_iter_num' + str(whole_iter_num) + '\n')
    fh.write(str(epoch) + '_epoch_total_loss' + epoch_total_loss + '\n')
    fh.write(str(epoch) + '_epoch_loss' + epoch_loss + '\n')
    fh.write('\n')
    fh.close()


def adjust_learning_rate(optimizer, decay_rate=.1):
    update_lr_group = optimizer.param_groups
    for param_group in update_lr_group:
        print('before lr: ', param_group['lr'])
        param_group['lr'] = param_group['lr'] * decay_rate
        print('after lr: ', param_group['lr'])
    return optimizer


def save_lr(save_dir, optimizer):
    update_lr_group = optimizer.param_groups[0]
    fh = open(save_dir, 'a')
    fh.write('encode:update:lr' + str(update_lr_group['lr']) + '\n')
    fh.write('decode:update:lr' + str(update_lr_group['lr']) + '\n')
    fh.write('\n')
    fh.close()


def train_net(num_gpus, args):

    mp.spawn(main, nprocs=num_gpus, args=(num_gpus, args))

def print_network(model):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    # print(model)
    print("The number of parameters: {}".format(num_params))


def LOG(output):
    with open('./loss.txt', 'a') as f:
        f.write(output)

def Val_test(test_loader, model, epoch, save_path, best_mae, best_epoch, args):
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        
        test_dataset = get_loader(test_loader, args.data_root, args.img_size, mode='test')
        test_loader = data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=1)
        
        for i, data_batch in enumerate(test_loader):
            image, depth, gt, image_w, image_h, image_path = data_batch
            
#             images, depths = Variable(images.cuda()), Variable(depths.cuda())
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            depth = depth.cuda()
            input = torch.cat((image, depth), dim=0)
            outputs_saliency, top_preds = model(input)
            mask_1_16, mask_1_8, mask_1_4, res = outputs_saliency
            
            _, _, GW, GH = gt.shape
            
            res = F.interpolate(res, size=(GW, GH), mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (GW * GH)
        mae = mae_sum / len(test_loader)

        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + args.mode + '_best.pth')

        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        LOG('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))
        return best_mae, best_epoch

def main(local_rank, num_gpus, args):

    cudnn.benchmark = True

    dist.init_process_group(backend='GLOO', init_method=args.init_method, world_size=num_gpus, rank=local_rank)

    torch.cuda.set_device(local_rank)

    net = ImageDepthNet(args)
    print_network(net)
    net.train()
    net.cuda()

    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = torch.nn.parallel.DistributedDataParallel(
        net,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True)

    base_params = [params for name, params in net.named_parameters() if ("backbone" in name)]
    other_params = [params for name, params in net.named_parameters() if ("backbone" not in name)]

    optimizer = optim.Adam([{'params': base_params, 'lr': args.lr * 0.1},
                            {'params': other_params, 'lr': args.lr}])
    train_dataset = get_loader(args.trainset, args.data_root, args.img_size, mode='train')

    sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=num_gpus,
        rank=local_rank,
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.works,
                                               pin_memory=True,
                                               sampler=sampler,
                                               drop_last=True,
                                               )

    print('''
        Starting training:
            Train steps: {}
            Batch size: {}
            Learning rate: {}
            Training size: {}
        '''.format(args.train_steps, args.batch_size, args.lr, len(train_loader.dataset)))

    N_train = len(train_loader) * args.batch_size

    loss_weights = [1, 0.8, 0.8, 0.5, 0.5, 0.5, 0.5]
    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)

    criterion = nn.BCEWithLogitsLoss()
    loss_function = torch.nn.CrossEntropyLoss()

    whole_iter_num = 0
    iter_num = math.ceil(len(train_loader.dataset) / args.batch_size)
    min_loss = 1
    best_mae = 1
    best_epoch = 0

    for epoch in range(args.epochs):

#         print('Starting epoch {}/{}.--lr:{}'.format(epoch + 1, args.epochs, args.lr))

        epoch_total_loss = 0
        epoch_loss = 0

        for i, data_batch in enumerate(train_loader):
            if (i + 1) > iter_num: break

            images, depths, label_224, label_14, label_28, label_56, label_112 = data_batch

            images, depths, label_224 = Variable(images.cuda(local_rank, non_blocking=True)), \
                                        Variable(depths.cuda(local_rank, non_blocking=True)), \
                                        Variable(label_224.cuda(local_rank, non_blocking=True))

            label_14, label_28, label_56, label_112  = Variable(label_14.cuda()), Variable(label_28.cuda()),\
                                                      Variable(label_56.cuda()), Variable(label_112.cuda())



            input = torch.cat((images, depths), dim=0)
            outputs_saliency, top_preds = net(input)
            mask_1_16, mask_1_8, mask_1_4 ,mask_1_1 = outputs_saliency
            toprgb, topdepth, toprgbd = top_preds


            # loss
            loss8 = criterion(toprgbd, label_14)
            loss7 = criterion(topdepth, label_14)
            loss6 = criterion(toprgb, label_14)
            loss5 = criterion(mask_1_16, label_14)
            loss4 = criterion(mask_1_8, label_28)
            loss2 = criterion(mask_1_4, label_56)
            loss1 = criterion(mask_1_1, label_224)



            img_total_loss = loss_weights[0] * loss1 + loss_weights[1] * loss2+ loss_weights[3] * loss4 + loss_weights[4] * loss5\
                             + loss_weights[4] * loss6+ loss_weights[4] * loss7+ loss_weights[4] * loss8


            total_loss = img_total_loss

            epoch_total_loss += total_loss.cpu().data.item()
            epoch_loss += loss1.cpu().data.item()

            optimizer.zero_grad()

            total_loss.backward()

            optimizer.step()
            whole_iter_num += 1

            if (local_rank == 0) and (whole_iter_num == args.train_steps):
                torch.save(net.state_dict(),
                           args.save_model_dir + args.mode +'.pth')

            if whole_iter_num == args.train_steps:
                return 0

            if whole_iter_num == args.stepvalue1 or whole_iter_num == args.stepvalue2:
                optimizer = adjust_learning_rate(optimizer, decay_rate=args.lr_decay_gamma)
                save_dir = './loss.txt'
                save_lr(save_dir, optimizer)
                print('have updated lr!!')
#         print('Epoch finished ! Loss: {}'.format(epoch_total_loss / iter_num))
        print('{} ------- Epoch {}, step: {}, Epoch_loss: {}'.format(datetime.now(), epoch+1, whole_iter_num, epoch_loss/iter_num))


        save_lossdir = './loss.txt'
        save_loss(save_lossdir, whole_iter_num, epoch_total_loss / iter_num, epoch_loss/iter_num, epoch+1)

        best_mae, best_epoch = Val_test(args.val_set, net, epoch+1, args.save_model_dir, best_mae, best_epoch, args)




