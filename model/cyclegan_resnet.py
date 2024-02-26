import os
import torch
import itertools

from model.layers import fn_helper as helper
from model.layers.loss_functions import GANLoss
from utils.events import LOGGER


class cycleforwardtrainer():
    def __init__(self, args, cfg, device):
        super(cycleforwardtrainer, self).__init__()
        self.args = args
        self.cfg = cfg
        self.device = device
        #
        self.istrain = self.args.istrain
        self.load_dir = self.args.save_dir + '/preprocessing_nextrained/'
        self.save_dir = self.args.save_dir + '/preprocessing_nextrained/'
        #
        self.metric = dict()

        if self.istrain:
            self.model_names = ['G_A', 'D_A']
        else:
            self.model_names = ['G_A']

        # define network for generator
        cfg_g = cfg.model_generators
        self.netG_A = helper.define_G(
            input_nc=cfg_g.input_nc, output_nc=cfg_g.output_nc, ngf=cfg_g.ngf,
            norm=cfg_g.norm, netGname=cfg_g.netGname, use_dropout=not cfg_g.no_dropout,
            init_type=cfg_g.init_type, init_gain=cfg_g.init_gain, device=self.device
        )

        if self.istrain:
            cfg_d = cfg.model_discriminator
            self.netD_B = helper.define_D(
                input_nc=cfg_d.output_nc, ndf=cfg_d.ndf, netDname=cfg_d.netDname,
                n_layer_D=cfg_d.n_layers_D, norm=cfg_d.norm,
                init_type=cfg_d.init_type, init_gain=cfg_d.init_gain, device=self.device
            )

            # optimizer
            self.optimizers = []
            if cfg.solver.optim == 'adam':
                self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters()), lr=cfg.solver.lr,
                                                    betas=(cfg.solver.beta, 0.999))
                self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_B.parameters()), lr=cfg.solver.lr,
                                                    betas=(cfg.solver.beta, 0.999))
            elif cfg.solver.optim == 'sgd':
                self.optimizer_G = torch.optim.SGD(itertools.chain(self.netG_A.parameters()),
                                                   lr=cfg.solver.lr, momentum=cfg.solver.momentum,
                                                   nesterov=True)
                self.optimizer_D = torch.optim.SGD(itertools.chain(self.netD_B.parameters()),
                                                   lr=cfg.solver.lr, momentum=cfg.solver.momentum,
                                                   nesterov=True)
            else:
                LOGGER.error('{} is an unusable optimizer'.format(cfg.solver.optim))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            # scheduler
            self.schedulers = []
            self.scheduler_G = helper.get_scheduler(self.optimizer_G, cfg.solver)
            self.scheduler_D = helper.get_scheduler(self.optimizer_D, cfg.solver)

            self.schedulers.append(self.scheduler_G)
            self.schedulers.append(self.scheduler_D)

            # loss function
            self.criterionGAN = GANLoss(args.gan_mode).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.MSE = torch.nn.MSELoss()

    def wdb_set_input(self, input):
        self.realA = input['A'].to(self.device)

    def set_input(self, input):
        self.realA = input['A'].to(self.device)
        if self.istrain:
            self.realB = input['B'].to(self.device)

    def forward(self):
        self.fakeB = self.netG_A(self.realA)  # G_A(A)
        return self.fakeB

    def run_generator(self):
        return self.forward()

    def calc_loss_G_A(self):
        # GAN loss D_B(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_B(self.fakeB), True)
        return self.loss_G_A

    def run_discriminator(self):
        self.loss_D_B = self.calc_loss_D(self.netD_B, self.realB, self.fakeB)

    def calc_loss_D(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        #
        return loss_D

    def update_discriminator(self):
        self.optimizer_D.zero_grad()
        self.loss_D_B.backward()
        self.optimizer_D.step()

    def update_learning_rate(self):
        for idx, scheduler in zip(range(2), self.schedulers):
            if self.cfg.solver.lr_scheduler == 'plateau':
                scheduler.step(self.metric[str(idx)])
            else:
                scheduler.step()
        print('learning rate = %.7f' % self.optimizer_G.param_groups[0]['lr'])

    def make_rec_image(self, fakeA):
        self.recB = self.netG_A(fakeA)

    def get_gen_output(self):
        return self.fakeB

    def get_fakeB(self):
        return self.fakeB

    def get_fakeC(self):
        return self.fakeC

    def get_recB(self):
        return self.recB

    def get_real_data(self):
        if self.istrain:
            return self.realA, self.realB
        else:
            return self.realA

    def get_loss_G_A(self):
        return self.loss_G_A

    def get_loss_D_B(self):
        return self.loss_D_B

    def save_networks(self, epoch='latest'):
        """Save all the networks to the disk.
        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_forward_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)
                torch.save(net.cpu().state_dict(), save_path)
                if len(self.args.gpuid) > 0 and torch.cuda.is_available():
                    net.to(self.device)

    def load_networks(self, epoch='latest'):
        """Load all the networks from the disk.
        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_forward_%s.pth' % (epoch, name)
                load_path = os.path.join(self.load_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)
                print('----------------------------------------------')
                print('[Forward Cycle {}] Successfully LOADED!'.format(name))
                print('----------------------------------------------')

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)


class cyclebackwardtrainer():
    def __init__(self, args, cfg, device):
        super(cyclebackwardtrainer, self).__init__()
        self.args = args
        self.cfg = cfg
        self.device = device
        #
        self.istrain = self.args.istrain
        self.load_dir = self.args.save_dir + '/preprocessing_nextrained/'
        self.save_dir = self.args.save_dir + '/preprocessing_nextrained/'
        #
        self.metric = dict()

        if self.istrain:
            self.model_names = ['G_B', 'D_B']
        else:
            self.model_names = ['G_B']

        # define network for generator
        cfg_g = cfg.model_generators
        self.netG_B = helper.define_G(
            input_nc=cfg_g.input_nc, output_nc=cfg_g.output_nc, ngf=cfg_g.ngf,
            norm=cfg_g.norm, netGname=cfg_g.netGname, use_dropout=not cfg_g.no_dropout,
            init_type=cfg_g.init_type, init_gain=cfg_g.init_gain, device=self.device
        )

        if self.istrain:
            cfg_d = cfg.model_discriminator
            self.netD_A = helper.define_D(
                input_nc=cfg_d.output_nc, ndf=cfg_d.ndf, netDname=cfg_d.netDname,
                n_layer_D=cfg_d.n_layers_D, norm=cfg_d.norm,
                init_type=cfg_d.init_type, init_gain=cfg_d.init_gain, device=self.device
            )

            # optimizer
            self.optimizers = []
            if cfg.solver.optim == 'adam':
                self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_B.parameters()), lr=cfg.solver.lr,
                                                    betas=(cfg.solver.beta, 0.999))
                self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters()), lr=cfg.solver.lr,
                                                    betas=(cfg.solver.beta, 0.999))
            elif cfg.solver.optim == 'sgd':
                self.optimizer_G = torch.optim.SGD(itertools.chain(self.netG_B.parameters()),
                                                   lr=cfg.solver.lr, momentum=cfg.solver.momentum,
                                                   nesterov=True)
                self.optimizer_D = torch.optim.SGD(itertools.chain(self.netD_A.parameters()),
                                                   lr=cfg.solver.lr, momentum=cfg.solver.momentum,
                                                   nesterov=True)
            else:
                LOGGER.error('{} is an unusable optimizer'.format(cfg.solver.optim))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            # scheduler
            self.schedulers = []
            self.scheduler_G = helper.get_scheduler(self.optimizer_G, cfg.solver)
            self.scheduler_D = helper.get_scheduler(self.optimizer_D, cfg.solver)

            self.schedulers.append(self.scheduler_G)
            self.schedulers.append(self.scheduler_D)
            # loss function
            self.criterionGAN = GANLoss(args.gan_mode).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()

    def set_input(self, input):
        self.realA = input['A'].to(self.device)
        self.realB = input['B'].to(self.device)

    def forward(self):
        self.fakeA = self.netG_B(self.realB)  # G_B(B)

    def run_generator(self):
        self.forward()

    def calc_loss_G_B(self):
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fakeA), True)
        return self.loss_G_A

    def run_discriminator(self):
        self.loss_D_A = self.calc_loss_D(self.netD_A, self.realA, self.fakeA)

    def calc_loss_D(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        #
        return loss_D

    def update_discriminator(self):
        self.optimizer_D.zero_grad()
        self.loss_D_A.backward()
        self.optimizer_D.step()

    def update_learning_rate(self):
        for idx, scheduler in zip(range(2), self.schedulers):
            if self.cfg.solver.lr_scheduler == 'plateau':
                scheduler.step(self.metric[str(idx)])
            else:
                scheduler.step()
        print('learning rate = %.7f' % self.optimizer_G.param_groups[0]['lr'])

    def make_rec_image(self, fakeB):
        self.recA = self.netG_B(fakeB)

    def get_gen_output(self):
        return self.fakeA

    def get_fakeA(self):
        return self.fakeA

    def get_recA(self):
        return self.recA

    def get_real_data(self):
        return self.realA, self.realB

    def get_loss_D_A(self):
        return self.loss_D_A

    def save_networks(self, epoch='latest'):
        """Save all the networks to the disk.
        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_backward_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)
                torch.save(net.cpu().state_dict(), save_path)
                if len(self.args.gpu_id) > 0 and torch.cuda.is_available():
                    net.to(self.device)

    def load_networks(self, epoch='latest'):
        """Load all the networks from the disk.
        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_backward_%s.pth' % (epoch, name)
                load_path = os.path.join(self.load_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)
                print('----------------------------------------------')
                print('[Backward Cycle {}] Successfully LOADED!'.format(name))
                print('----------------------------------------------')

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)
