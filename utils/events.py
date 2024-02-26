import os
import shutil
import yaml
import logging


def set_logging(name=None):
    rank = int(os.getenv('RANK', -1))
    logging.basicConfig(format="%(message)s", level=logging.INFO if (rank in (-1, 0)) else logging.WARNING)
    return logging.getLogger(name)


LOGGER = set_logging(__name__)
NCOLS = min(100, shutil.get_terminal_size().columns)


def load_yaml(file_path):
    """Load data from yaml file."""
    if isinstance(file_path, str):
        with open(file_path, errors='ignore') as f:
            data_dict = yaml.safe_load(f)
    return data_dict


def write_tbloss(tblogger, loss_dict, step):
    tblogger.add_scalars("training/loss/model", {
        'loss_G_A': loss_dict['loss_G_A'][-1],
        'loss_G_B': loss_dict['loss_G_B'][-1],
        'loss_cycle_A': loss_dict['loss_cycle_A'][-1],
        'loss_cycle_B': loss_dict['loss_cycle_B'][-1],
        'loss_D_A': loss_dict['loss_D_A'][-1],
        'loss_D_B': loss_dict['loss_D_B'][-1],
        'total_loss': loss_dict['total_loss'][-1],
    }, step + 1)


def write_tbimg(tblogger, imgs, step, type='train'):
    """Display train_batch and validation predictions to tensorboard."""
    if type == 'train':
        tblogger.add_image(f'train_batch', imgs, step + 1, dataformats='HWC')
    elif type == 'val':
        for idx, img in enumerate(imgs):
            tblogger.add_image(f'val_img_{idx + 1}', img, step + 1, dataformats='HWC')
    else:
        LOGGER.warning('WARNING: Unknown image type to visualize.\n')


def write_tbloss_box(tblogger, loss_dict, step):
    tblogger.add_scalar("train/loss_G_A", loss_dict['loss_G_A'][-1], step + 1)
    tblogger.add_scalar("train/loss_G_B", loss_dict['loss_G_B'][-1], step + 1)
    tblogger.add_scalar("train/loss_cycle_A", loss_dict['loss_cycle_A'][-1], step + 1)
    tblogger.add_scalar("train/loss_cycle_B", loss_dict['loss_cycle_B'][-1], step + 1)
    tblogger.add_scalar("train/loss_D_A", loss_dict['loss_D_A'][-1], step + 1)
    tblogger.add_scalar("train/loss_D_B", loss_dict['loss_D_B'][-1], step + 1)
    tblogger.add_scalar("train/total_loss", loss_dict['total_loss'][-1], step + 1)
