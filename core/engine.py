import os
import time
import torch
import numpy as np

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from dataloader.fisheyes_dataloader import FishEyesDataset
from model.cyclegan_resnet import cycleforwardtrainer, cyclebackwardtrainer
from utils.events import LOGGER, NCOLS, load_yaml, write_tbimg, write_tbloss_box


class Trainer:
    def __init__(self, args, cfg, device):
        self.args = args
        self.cfg = cfg
        self.device = device
        self.save_dir = args.save_dir
        #
        self.data_dict = load_yaml(args.data_path)
        #
        self.train_loader = self.get_data_loader(self.data_dict)
        #
        self.model = self.get_model(args, cfg, device)
        #
        self.tblogger = SummaryWriter(os.path.join(self.save_dir, 'logging'), 'logging')
        #
        self.start_epoch = 0
        self.max_epoch = args.epoch
        self.max_stepnum = len(self.train_loader)

    def train(self):
        self.start_time = time.time()

        try:
            LOGGER.info(f'Training start...')
            for self.epoch in range(self.start_epoch, self.max_epoch):
                self.train_one_epoch(self.epoch)
            LOGGER.info(f'\nTraining completed in {(time.time() - self.start_time) / 3600:.3f} hours.')
            LOGGER.info(f'Strip optimizer from the saved pt model...')
        except Exception as _:
            LOGGER.error('ERROR in training loop or eval/save model.')
            raise
        finally:
            if self.device != 'cpu':
                torch.cuda.empty_cache()

    def train_one_epoch(self, epoch_num):
        try:
            self.pbar = tqdm(self.train_loader, total=self.max_stepnum, ncols=NCOLS,
                             bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

            loss_dict = dict()
            loss_dict['loss_G_A'], loss_dict['loss_G_B'] = [], []
            loss_dict['loss_cycle_A'], loss_dict['loss_cycle_B'] = [], []
            loss_dict['loss_D_A'], loss_dict['loss_D_B'] = [0], [0]
            loss_dict['loss_mse'] = []
            loss_dict['total_loss'] = []
            for self.step, self.batch_data in enumerate(self.pbar):
                input_dict = dict()

                input_dict['A'] = self.batch_data[0].to(self.device, non_blocking=True).float() / 255
                input_dict['B'] = self.batch_data[1].to(self.device, non_blocking=True).float() / 255

                self.model['forward'].set_input(input_dict)
                self.model['backward'].set_input(input_dict)

                # forward
                self.model['forward'].run_generator()

                # backward
                self.model['backward'].run_generator()

                # reconst
                self.model['forward'].make_rec_image(self.model['backward'].get_fakeA())
                self.model['backward'].make_rec_image(self.model['forward'].get_fakeB())

                # calculate loss
                lambA = self.cfg.loss_param.lambda_A
                lambB = self.cfg.loss_param.lambda_B
                lambC = self.cfg.loss_param.lambda_C
                lambD = self.cfg.loss_param.lambda_D

                recB = self.model['forward'].get_recB()
                recA = self.model['backward'].get_recA()
                realA, realB = self.model['forward'].get_real_data()

                fakeB = self.model['forward'].get_fakeB()
                fakeA = self.model['backward'].get_fakeA()
                fakeB_avg = self.function_A(fakeB)

                loss_G_A = self.model['forward'].calc_loss_G_A()
                loss_G_B = self.model['backward'].calc_loss_G_B()
                loss_cycle_A = self.model['forward'].criterionCycle(recA, realA) * lambA
                loss_cycle_B = self.model['forward'].criterionCycle(recB, realB) * lambB
                # loss_L1_G_A = self.model['forward'].criterionCycle(fakeB, realB) * lambC
                # loss_mse = self.model['forward'].MSE(fakeB_avg, input_dict['C']) * lambD
                # --- Losses of Discriminators for forward and backward

                # total loss of generator
                loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B

                loss_dict['loss_G_A'].append(loss_G_A), loss_dict['loss_G_B'].append(loss_G_B)
                loss_dict['loss_cycle_A'].append(loss_cycle_A), loss_dict['loss_cycle_B'].append(loss_cycle_B)
                loss_dict['total_loss'].append(loss_G)

                # update
                self.model['forward'].optimizer_G.zero_grad()
                self.model['backward'].optimizer_G.zero_grad()
                loss_G.backward()
                self.model['forward'].optimizer_G.step()
                self.model['backward'].optimizer_G.step()

                if (self.step % self.args.update_discriminator_step) == 0:
                    self.model['forward'].run_discriminator()
                    self.model['backward'].run_discriminator()

                    self.model['forward'].update_discriminator()
                    self.model['backward'].update_discriminator()

                    loss_D_B = self.model['forward'].get_loss_D_B()
                    loss_D_A = self.model['backward'].get_loss_D_A()

                    loss_dict['loss_D_A'].append(loss_D_A), loss_dict['loss_D_B'].append(loss_D_B)

                self.pbar.set_postfix(loss=round(loss_G.item(), 2))

                if (self.step % self.args.show_tb) == 0:
                    vis_train_batch = self.plot_train_batch([realA, fakeA, realB, fakeB])
                    self.tblogger.add_image('realA', vis_train_batch[0], self.step + 1, dataformats='HWC')
                    self.tblogger.add_image('fakeA', vis_train_batch[1], self.step + 1, dataformats='HWC')
                    self.tblogger.add_image('realB', vis_train_batch[2], self.step + 1, dataformats='HWC')
                    self.tblogger.add_image('fakeB', vis_train_batch[3], self.step + 1, dataformats='HWC')

                    write_tbloss_box(self.tblogger, loss_dict=loss_dict,
                                     step=(self.epoch * self.max_stepnum) + self.step)

        except Exception as _:
            LOGGER.error('ERROR in training steps')
            raise

    def get_data_loader(self, data_dict):
        dataset_object_train = FishEyesDataset(
            data_dict,
            img_size=self.args.img_size,
            hyp=dict(self.cfg.data_aug),
            augment=True,
            input_gray=self.args.input_gray
        )
        train_loader = torch.utils.data.DataLoader(
            dataset=dataset_object_train,
            batch_size=self.args.batch,
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            collate_fn=dataset_object_train.collate_fn,
        )

        return train_loader

    def get_model(self, args, cfg, device):
        net_dict = dict()
        #
        net_dict['forward'] = cycleforwardtrainer(args, cfg, device)
        net_dict['backward'] = cyclebackwardtrainer(args, cfg, device)
        #
        return net_dict

    @staticmethod
    def function_A(fake_img):
        fake_img_avg = []
        for i in range(fake_img.shape[0]):
            f = fake_img[i].squeeze(0)
            ch_img = torch.abs(f - torch.full_like(f, float(torch.max(f))))
            fake_img_avg.append(float(torch.mean(ch_img[ch_img > float(torch.max(ch_img)) * 0.9])))
        return torch.tensor(fake_img_avg)

    def plot_train_batch(self, img_list):
        new_img_list = []
        for img in img_list:
            new_img = img[0]
            if isinstance(img, torch.Tensor):
                new_img = new_img.permute(1, 2, 0).detach().cpu().float().numpy()
            if np.max(new_img) <= 1:
                new_img *= 255  # de-normalise (optional)
            new_img_list.append(new_img.astype(np.uint8))
        return new_img_list

