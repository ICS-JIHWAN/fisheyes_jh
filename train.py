import os
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader

from cmd_in import get_args_parser
from utils.metrics import gaussuian_filter
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

    # Data loader
    dataset_obj = Dataset(
        image_dir=STORAGE,
        image_size=args.img_size,
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
    G = Generator().to(device)
    D = Discriminator().to(device)
    # segmenter
    segm_cfg = dict(
        encoder=dict(
            image_size=args.image_size,
            patch_size=args.patch_size,
            n_layers=24,
            d_model=1024,
            d_ff=4*1024,        # 4 * d_model
            n_heads=16,
            n_cls=args.n_cls,
        ),
        decoder=dict(
            n_cls=args.n_cls,
            patch_size=args.patch_size,
            d_encoder=1024,     # encoder.d_model
            n_layers=1,
            n_heads=1024//64,   # encoder.d_model//64
            d_model=1024,       # encoder.d_model
            d_ff=4*1024,        # 4 * encoder.d_model
            drop_path_rate=0.0,
            dropout=0.1,
        )
    )
    encoder = VisionTransformer(**segm_cfg['encoder'])
    decoder = MaskTransformer(**segm_cfg['decoder'])
    Segm = Segmenter(encoder, decoder, args.n_cls)

    # Loss
    criterionL1 = nn.L1Loss().to(device)
    criterionMSE = nn.MSELoss().to(device)

    # Optimizer
    g_optimizer = optim.Adam(G.parameters(), lr=args.lr, betas=args.betas)
    d_optimizer = optim.Adam(D.parameters(), lr=args.lr, betas=args.betas)

    # Train
    for epoch in range(args.epoch):
        print('\033[31m' + f'Train Epoch Nums : {epoch + 1} / {args.epoch}' + '\033[0m')
        pbar = tqdm(train_loader)
        for i, (real_source, real_target) in enumerate(pbar):
            # Image load
            real_source, real_target = real_source.to(device), real_target.to(device)

            # Label
            B, C, H, W = real_source.shape
            real_label = torch.ones((B, 1, H // (2 ** 3) - 2, W // (2 ** 3) - 2)).to(device)
            fake_label = torch.zeros((B, 1, H // (2 ** 3) - 2, W // (2 ** 3) - 2)).to(device)

            # Source -> Target
            fake_target = G.forward(real_source)

            # =============== Train the generator =============== #
            # G(A) should fake the discriminator
            pred_fake = D.forward(fake_target)
            #
            loss_g_gan = criterionMSE(pred_fake, real_label)

            # L1 loss 를 위한 Gaussian weight 적용
            gauss_weight = torch.tensor(gaussuian_filter(args.img_size, sigma=0.5)).to(device)
            gauss_weight = gauss_weight ** 2

            loss_g_l1 = criterionL1(fake_target * (1 - gauss_weight), real_target * (1 - gauss_weight))  # fisheye 모양
            loss_g_l1_s = criterionL1(fake_target * gauss_weight, real_source * gauss_weight)  # 원본 source 모양

            # Generator loss
            loss_g = loss_g_gan + loss_g_l1 * 0.07 + loss_g_l1_s * 0.4

            # Generator update
            G.zero_grad()
            D.zero_grad()
            loss_g.backward()
            g_optimizer.step()

            # ============= Train the discriminator ============= #
            # train with fake
            pred_fake = D.forward(fake_target.detach())
            loss_d_fake = criterionMSE(pred_fake, fake_label)

            # train with real
            pred_real = D.forward(real_target)
            loss_d_real = criterionMSE(pred_real, real_label)

            # Discriminator loss
            loss_d = (loss_d_fake + loss_d_real) * 0.5

            # Discriminator update
            D.zero_grad()
            loss_d.backward()
            d_optimizer.step()


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
