import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from data import aux_dataset
from . import networks


class AttnCycleGANV2Model(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        parser.add_argument('--mask_size', type=int, default=128)
        parser.add_argument('--s1', type=int, default=32)
        parser.add_argument('--s2', type=int, default=16)
        parser.add_argument('--concat', type=str, default='rmult')
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.0, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--use_rec', action='store_true', help='Whether using the reconstructed image to update the discriminator')
            parser.add_argument('--use_warmup', action='store_true', help='Whether using the warm up phase to stablize the discriminator')
            parser.add_argument('--warmup_epoch', type=int, default=30)
            parser.add_argument('--rec_coef', type=float, default=.5, help='the coefficient of x'' loss')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B

        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']

        if opt.concat != 'alpha':
            self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        else:
            self.netG_A = networks.define_G(4, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netG_B = networks.define_G(4, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # Additional attention map pool for x~G(x) and G(x)~G'(G(x))
        # self.aux_data = aux_dataset.AuxAttnDataset(3000, 3000, self.gpu_ids[0], mask_size=opt.mask_size)
        # self.aux_gen_data = aux_dataset.AuxAttnDataset(3000, 3000, self.gpu_ids[0], mask_size=opt.mask_size)

        self.zero_attn_holder = torch.zeros((1, 1, opt.mask_size, opt.mask_size), dtype=torch.float32).to(self.device)
        self.ones_attn_holder = torch.ones((1, 1, opt.mask_size, opt.mask_size), dtype=torch.float32).to(self.device)

        self.concat = opt.concat

        if True:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids,
                                            opt.mask_size, opt.s1, opt.s2)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids,
                                            opt.mask_size, opt.s1, opt.s2)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                # assert(opt.input_nc == opt.output_nc)
                pass
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.use_rec = opt.use_rec
            self.rec_coef = opt.rec_coef

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)

        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        '''Reminder: D_A handles domain B and D_B handles domain A'''
        _, attn_A = self.netD_B(self.real_A)
        _, attn_B = self.netD_A(self.real_B)

        self.temp_attn_A, self.temp_attn_B = attn_A, attn_B

        if self.concat == 'alpha':
            self.fake_B = self.netG_A(torch.cat((self.real_A, attn_A.detach()), 1))  # X -> G(X) = Y'
            self.fake_A = self.netG_B(torch.cat((self.real_B, attn_B.detach()), 1))  # Y -> G(Y) = X'

        elif self.concat == 'mult':
            self.fake_B = self.netG_A(self.real_A * attn_A.detach())
            self.fake_A = self.netG_B(self.real_B * attn_B.detach())

        elif self.concat == 'rmult':
            self.fake_B = self.netG_A(self.real_A * (1. + attn_A.detach()))
            self.fake_A = self.netG_B(self.real_B * (1. + attn_B.detach()))

        elif self.concat == 'none':
            self.fake_B = self.netG_A(self.real_A)  # G_A(A)
            self.fake_A = self.netG_B(self.real_B)  # G_B(B)

        else:
            raise NotImplementedError('Unsupported concatenation operation')

        _, attn_fake_B = self.netD_A(self.fake_B)
        _, attn_fake_A = self.netD_B(self.fake_A)

        if self.concat == 'alpha':
            self.rec_A = self.netG_B(torch.cat((self.fake_B, attn_fake_B.detach()), 1))
            self.rec_B = self.netG_A(torch.cat((self.fake_A, attn_fake_A.detach()), 1))

        elif self.concat == 'mult':
            self.rec_A = self.netG_B(self.fake_B * attn_fake_B.detach())
            self.rec_B = self.netG_A(self.fake_A * attn_fake_A.detach())

        elif self.concat == 'rmult':
            self.rec_A = self.netG_B(self.fake_B * (1. + attn_fake_B.detach()))
            self.rec_B = self.netG_A(self.fake_A * (1. + attn_fake_A.detach()))

        elif self.concat == 'none':
            self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A))
            self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B))

        else:
            raise NotImplementedError('Unsupported concatenation operation')

    def forward_test(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.concat == 'alpha':
            self.fake_B = self.netG_A(torch.cat((self.real_A, self.attn_A), 1))
            self.rec_A = self.netG_B(torch.cat((self.fake_B, self.ones_attn_holder), 1))
            self.fake_A = self.netG_B(torch.cat((self.real_B, self.attn_B), 1))
            self.rec_B = self.netG_A(torch.cat((self.fake_A, self.ones_attn_holder), 1))
        elif self.concat == 'mult':
            self.fake_B = self.netG_A(self.real_A * self.attn_A)
            self.rec_A = self.netG_B(self.fake_B)
            self.fake_A = self.netG_B(self.real_B * self.attn_B)
            self.rec_B = self.netG_A(self.fake_A)
        elif self.concat == 'rmult':
            self.fake_B = self.netG_A(self.real_A * 1.0)
            self.rec_A = self.netG_B(self.fake_B)
            self.fake_A = self.netG_B(self.real_B * 1.0)
            self.rec_B = self.netG_A(self.fake_A)
        elif self.concat == 'none':
            self.fake_B = self.netG_A(self.real_A)  # G_A(A)
            self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A))
            self.fake_A = self.netG_B(self.real_B)  # G_B(B)
            self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B))
        else:
            raise NotImplementedError('Unsupported concatenation operation')

    def backward_D_basic(self, netD, real, fake, extra=None):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator
            extra_fake (tensor array) -- real images from another domain

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real, _ = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)

        # Fake
        pred_fake, _ = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)

        loss_D = (loss_D_real + loss_D_fake) * 0.5

        # Extra rec update
        if extra is not None:
            pred_extra, _ = netD(extra.detach())
            loss_D_extra = self.criterionGAN(pred_extra, False)
            loss_D += self.rec_coef * loss_D_extra

        # Combined loss and calculate gradients
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        # self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B, extra=self.rec_B if self.use_rec else None)
        # test = self.backward_D_basic(self.netD_A, self.real_B, )

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        # self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A, extra=self.rec_A if self.use_rec else None)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            if self.concat == 'alpha':
                self.idt_A = self.netG_A(torch.cat((self.real_B, self.temp_attn_B), 1))
                self.idt_B = self.netG_B(torch.cat((self.real_A, self.temp_attn_A), 1))
            elif self.concat == 'rmult':
                self.idt_A = self.netG_A(self.real_B * (1. + self.temp_attn_B))
                self.idt_B = self.netG_B(self.real_A * (1. + self.temp_attn_A))
            elif self.concat == 'mult':
                self.idt_A = self.netG_A(self.real_B * (0. + self.temp_attn_B))
                self.idt_B = self.netG_B(self.real_A * (0. + self.temp_attn_A))
            else:
                self.idt_A = self.netG_A(self.real_B)
                self.idt_B = self.netG_B(self.real_A)

            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        dis_A_res, _ = self.netD_A(self.fake_B)
        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(dis_A_res, True)

        dis_B_res, _ = self.netD_B(self.fake_A)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(dis_B_res, True)

        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()