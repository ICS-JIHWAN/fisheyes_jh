import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from cmd_in import get_args_parser
from utils.metrics import gaussuian_filter, create_color_map, flow_warp
from utils.events import save_yaml
from dataset.dataloader import Dataset
from model.patchgan import Generator, Discriminator
from model.segmenter import VisionTransformer, MaskTransformer, Segmenter

STORAGE = '/storage/jhchoi/fish_eyes'


def main(args, device):
    #
    # Save tensorboard & check point
    if args.save:
        from torch.utils.tensorboard import SummaryWriter
        save_path = os.path.join(STORAGE, args.save_dir)
        tblogger = SummaryWriter(os.path.join(save_path, 'logging'))
        save_yaml(vars(args), os.path.join(save_path, 'variant.yml'))

    # Data loader
    dataset_obj = Dataset(
        image_dir=STORAGE,
        image_size=args.img_size,
        n_cls=args.n_cls,
        norm=args.normalization,
    )
    train_loader = DataLoader(
        dataset=dataset_obj,
        num_workers=args.num_workers,
        batch_size=args.batch,
        shuffle=True
    )

    # Model
    # patchgan
    G = Generator(out_ch=2).to(device)
    D = Discriminator(n_cls=args.n_cls).to(device)

    if args.resume:
        load_model(G, torch.load(os.path.join(STORAGE, args.check_gan, 'checkpoint.pth'), map_location=device)['G'])
        load_model(D, torch.load(os.path.join(STORAGE, args.check_gan, 'checkpoint.pth'), map_location=device)['D'])

    # segmenter
    segm_cfg = dict(
        encoder=dict(
            image_size=(args.img_size, args.img_size),
            patch_size=args.patch_size,
            n_layers=24,
            d_model=1024,
            d_ff=4 * 1024,  # 4 * d_model
            n_heads=16,
            n_cls=args.n_cls,
        ),
        decoder=dict(
            n_cls=args.n_cls,
            patch_size=args.patch_size,
            d_encoder=1024,  # encoder.d_model
            n_layers=1,
            n_heads=1024 // 64,  # encoder.d_model//64
            d_model=1024,  # encoder.d_model
            d_ff=4 * 1024,  # 4 * encoder.d_model
            drop_path_rate=0.0,
            dropout=0.1,
        )
    )
    encoder = VisionTransformer(**segm_cfg['encoder'])
    decoder = MaskTransformer(**segm_cfg['decoder'])
    Segm = Segmenter(encoder, decoder, args.n_cls).to(device)

    load_model(Segm, torch.load(os.path.join(STORAGE, args.check_segm, 'checkpoint.pth'), map_location=device)['model'])

    # Loss
    criterionL1 = nn.L1Loss().to(device)
    criterionMSE = nn.MSELoss().to(device)

    # Optimizer
    g_optimizer = optim.Adam(G.parameters(), lr=args.lr, betas=args.betas)
    d_optimizer = optim.Adam(D.parameters(), lr=args.lr, betas=args.betas)

    g_scheduler = optim.lr_scheduler.CosineAnnealingLR(g_optimizer, T_max=20, eta_min=1e-6)
    d_scheduler = optim.lr_scheduler.CosineAnnealingLR(d_optimizer, T_max=20, eta_min=1e-5)

    # Train
    for epoch in range(args.epoch):
        print('\033[31m' + f'Train Epoch Nums : {epoch + 1} / {args.epoch}' + '\033[0m')
        pbar = tqdm(train_loader)
        for i, (real_source, real_target, real_gt, ignore_mask) in enumerate(pbar):
            # Image load
            real_source, real_target, real_gt = real_source.to(device), real_target.to(device), real_gt.to(device)

            # Label
            B, C, H, W = real_source.shape
            real_label = torch.ones((B, 1, H // (2 ** 3) - 2, W // (2 ** 3) - 2)).to(device)
            fake_label = torch.zeros((B, 1, H // (2 ** 3) - 2, W // (2 ** 3) - 2)).to(device)

            # Source -> Target
            pred_ru = G.forward(real_source)  # S ->[G]->F_S
            fake_target = flow_warp(real_source, pred_ru.permute(0, 2, 3, 1), padding_mode='zeros')

            # Segmentation
            with torch.no_grad():
                real_source_m = Segm.forward(real_source)
                real_target_m = Segm.forward(real_target)
                fake_target_m = Segm.forward(fake_target)

            # =============== Train the generator =============== #
            # G(A) should fake the discriminator
            # pred_fake = D.forward(fake_target)
            pred_fake = D.forward(torch.cat((fake_target, fake_target_m), dim=1))
            #
            loss_g_gan = criterionMSE(pred_fake, real_label)

            # L1 loss 를 위한 Gaussian weight 적용
            gauss_weight = torch.tensor(gaussuian_filter(args.img_size, sigma=0.5)).to(device)
            gauss_weight = gauss_weight ** 2

            loss_g_l1_t = criterionL1(fake_target * (1 - gauss_weight), real_target * (1 - gauss_weight))  # fisheye 모양
            loss_g_l1_s = criterionL1(fake_target * gauss_weight, real_source * gauss_weight)  # 원본 source 모양
            loss_g_l1_sm = criterionL1(fake_target_m * gauss_weight, real_source_m * gauss_weight)  # source semantic loss
            loss_g_l1_tm = criterionL1(real_target_m * (1 - gauss_weight), fake_target_m * (1 - gauss_weight))  # target semantic loss

            # Generator loss
            loss_g = loss_g_gan + loss_g_l1_t * 0.04 + loss_g_l1_s * 0.7 + loss_g_l1_sm * 0.04 + loss_g_l1_tm * 0.01

            # Generator update
            G.zero_grad()
            D.zero_grad()
            loss_g.backward()
            g_optimizer.step()

            # ============= Train the discriminator ============= #
            # train with fake
            # pred_fake = D.forward(fake_target.detach())
            pred_fake = D.forward(torch.cat((fake_target, fake_target_m), dim=1).detach())
            loss_d_fake = criterionMSE(pred_fake, fake_label)

            # train with real
            pred_real = D.forward(torch.cat((real_target, real_target_m), dim=1))
            loss_d_real = criterionMSE(pred_real, real_label)

            # Discriminator loss
            loss_d = (loss_d_fake + loss_d_real) * 0.5

            # Discriminator update
            D.zero_grad()
            loss_d.backward()
            d_optimizer.step()

            pbar.set_postfix(loss_g=round(loss_g.item(), 2), loss_d=round(loss_d.item(), 2))

            if args.save:
                if i % args.tb_show == 0:
                    # normalize 방식에 따라 다르게 normalize 를 풀어 줘야함
                    if args.normalization == 'vit':
                        invTrans = transforms.Compose(
                            [transforms.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                             transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]), ]
                        )

                        vis_real_source = invTrans(real_source)
                        vis_fake_target = invTrans(fake_target)
                        vis_real_target = invTrans(real_target)
                    else:
                        vis_real_source = denorm(real_source)
                        vis_fake_target = denorm(fake_target)
                        vis_real_target = denorm(real_target)

                    vis_real_source = torchvision.utils.make_grid(vis_real_source)[[2, 1, 0], ...]
                    vis_fake_target = torchvision.utils.make_grid(vis_fake_target)[[2, 1, 0], ...]
                    vis_real_target = torchvision.utils.make_grid(vis_real_target)[[2, 1, 0], ...]
                    vis_real_source_m = torchvision.utils.make_grid(create_color_map(real_source_m))[[2, 1, 0], ...]
                    vis_fake_target_m = torchvision.utils.make_grid(create_color_map(fake_target_m))[[2, 1, 0], ...]
                    vis_real_gt = torchvision.utils.make_grid(create_color_map(real_gt))[[2, 1, 0], ...]

                    tblogger.add_image('Image/Is', vis_real_source, epoch * 10000 + i)
                    tblogger.add_image('Image/Ist', vis_fake_target, epoch * 10000 + i)
                    tblogger.add_image('Image/It', vis_real_target, epoch * 10000 + i)

                    tblogger.add_image('Mask/Is_m', vis_real_source_m.type(torch.uint8), epoch * 10000 + i)
                    tblogger.add_image('Mask/Ist_m', vis_fake_target_m.type(torch.uint8), epoch * 10000 + i)
                    tblogger.add_image('Mask/Igt', vis_real_gt.type(torch.uint8), epoch * 10000 + i)

        d_scheduler.step()
        g_scheduler.step()
        if args.save:
            snapshot = dict(
                G=G.state_dict(),
                D=D.state_dict()
            )
            torch.save(snapshot, os.path.join(save_path, 'checkpoint.pth'))



def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


def load_model(model, checkpoint):
    try:
        model.load_state_dict(checkpoint, strict=False)
    except:
        print('Checking key between my model and load model')
        missmatch_keys = []
        for key in checkpoint.keys():
            if model.get_parameter(key).shape != checkpoint[key].shape:
                missmatch_keys.append(key)

        for mk in missmatch_keys:
            checkpoint.pop(mk)

        print(f'Missmatch size of keys are {missmatch_keys}')
        del missmatch_keys

        model.load_state_dict(checkpoint, strict=False)

    print('Model weights loaded successfully')


if __name__ == '__main__':
    #
    args = get_args_parser().parse_args()
    #
    if args.gpu_id is not None:
        device = torch.device('cuda:{}'.format(args.gpu_id))
        torch.cuda.set_device(args.gpu_id)
    else:
        device = torch.device('cpu')
    #
    main(args, device)
