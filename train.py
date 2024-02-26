import torch

from utils.config import Config
from cmd_in import get_args_parser
from utils.events import LOGGER
from core.engine import Trainer

if __name__ == '__main__':
    torch.cuda.set_device(1)
    #
    args = get_args_parser().parse_args()
    cfg = Config.fromfile(args.conf_file)
    LOGGER.info(f'training args are: {args}\n')

    # Setup
    from setproctitle import *
    setproctitle(args.procname)

    # Device
    device = torch.device('cuda:{}'.format(args.gpu_id)) if args.gpu_id is not None else torch.device('cpu')

    # Get trainer
    trainer = Trainer(args, cfg, device)

    # Start training
    trainer.train()
