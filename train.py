import torch

from config.config_utils import Config
from dataloader.fisheyes import FishEyesDataset
from utils.events import LOGGER, load_yaml
from cmd_in import get_args_parser


def train(args, cfg, device):
    data_dict = load_yaml(args.data_path)

    #
    dataset_object_train = FishEyesDataset(
        data_dict,
        img_size=args.img_size,
        n_cls=args.n_cls,
        check_path=args.check_path,
        normalization=args.normalization
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset_object_train,
        batch_size=args.batch,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last=False,
        collate_fn=dataset_object_train.collate_fn,
    )

    #
    from model.factory import create_G, create_D, create_Seg

    model_G   = create_G(cfg.model['GAN'])
    model_D   = create_D(cfg.model['GAN'])
    model_Seg = create_Seg(cfg.model['segmentation'], args, enc_name=args.encoder_name)



if __name__ == '__main__':
    #
    args = get_args_parser().parse_args()
    cfg = Config.fromfile(args.conf_file)
    LOGGER.info(f'training args are: {args}\n')

    # Setup
    from setproctitle import *

    setproctitle(args.procname)

    # Device
    device = torch.device('cuda:{}'.format(args.gpu_id)) if args.gpu_id is not None else torch.device('cpu')
    torch.cuda.set_device(args.gpu_id)

    train(args, cfg, device)
