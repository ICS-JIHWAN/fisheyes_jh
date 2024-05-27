# encoder : ViT
model = dict(
    GAN=dict(
        generator=dict(
            num_blocks=[2, 2, 2, 2]  # last fcn out
        ),

        discriminator=dict(
            input_nc=4,
            ndf=64,
            netDname='pixel',
            n_layers_D=3,
            norm='batch',
            init_type='xavier',
            init_gain=0.02
        )
    ),

    segmentation=dict(
        encoder=dict(
            vit_base_path8_384=dict(
                d_model=768,
                n_heads=12,
                n_layers=12,
                normalization='vit',
                distilled=False
            ),
            vit_large_patch16_384=dict(
                d_model=1024,
                n_heads=16,
                n_layers=24,
                normalization='vit'
            )
        ),
        decoder=dict(
            name='mask_transformer',
            drop_path_rate=0.0,
            dropout=0.1,
            n_layers=1
        )
    ),
)

solver = dict(
    opt='sgd',
    sched='polynomial',
    lr=1e-3,
    lrf=0.1,
    momentum=0.9,
    weight_decay=5e-4,
    min_lr=1e-5,
    poly_power=0.9,
    poly_step_size=1
)
