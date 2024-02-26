model_generators = dict(
    input_nc=1,                 # The number of input image channels
    output_nc=1,                # The number of output image channels
    ngf=64,                     # The number of generator filters in the last conv layer
    norm='batch',               # Normalization [instance|batch|none]
    netGname='resnet_9blocks',  # Specify generator architecture [resnet_9blocks]
    no_dropout=True,            # No dropout for generator
    init_type='xavier',         # Network initialization [normal|xavier|kaiming|orthogonal]
    init_gain=0.02,             # Scaling factor for normal, xavier and orthogonal
)

model_discriminator = dict(
    output_nc=1,
    ndf=64,
    netDname='n_layers',
    n_layers_D=3,
    norm='batch',
    init_type='xavier',
    init_gain=0.02,
)

loss_param = dict(
    lambda_A=10.0,
    lambda_B=10.0,
    lambda_C=10.0,
    lambda_D=10.0,
)

solver = dict(
    optim='adam',
    lr_scheduler='cosine',
    lr=1e-3,
    lr_min=1e-8,
    beta=0.5,
    lr_decay_iters=50,
    sch_niter=50,
    momentum=0.937,
    weight_decay=5e-4,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1
)

data_aug = dict(
    use_letterbox=False,
    letterbox_color=(0, 0, 0),
    letterbox_return_int=False
)
