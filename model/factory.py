from model.generators_jh import ResnetGenerator
from model.segmenter import Segmenter
from model.layers import fn_helper as helper
from model.layers.vit import VisionTransformer
from model.layers.decoder import DecoderLinear, MaskTransformer


def create_G(model_cfg):
    model_cfg = model_cfg['generator'].copy()

    model = ResnetGenerator(**model_cfg)

    return model


def create_D(model_cfg):
    model_cfg = model_cfg['discriminator']

    model = helper.define_D(**model_cfg)

    return model


def create_Seg(model_cfg, args, enc_name='vit_large_patch16_384'):

    encoder = create_vit(model_cfg['encoder'][enc_name], args)
    decoder = create_decoder(encoder, model_cfg['decoder'], args)

    model = Segmenter(encoder, decoder, n_cls=args.n_cls)

    return model


def create_vit(model_cfg, args):
    model_cfg = model_cfg.copy()

    model_cfg['n_cls'] = args.n_cls
    model_cfg['image_size'] = (args.img_size, args.img_size)
    model_cfg['patch_size'] = args.patch_size

    model_cfg.pop("normalization")

    mlp_expansion_ratio = 4
    model_cfg["d_ff"] = mlp_expansion_ratio * model_cfg["d_model"]

    model = VisionTransformer(**model_cfg)

    return model


def create_decoder(encoder, model_cfg, args):
    decoder_cfg = model_cfg['decoder'].copy()
    name = decoder_cfg.pop("name")
    decoder_cfg["n_cls"] = args.n_cls
    decoder_cfg["d_encoder"] = encoder.d_model
    decoder_cfg["patch_size"] = encoder.patch_size

    if "linear" in name:
        decoder = DecoderLinear(**decoder_cfg)
    elif name == "mask_transformer":
        dim = encoder.d_model
        n_heads = dim // 64
        decoder_cfg["n_heads"] = n_heads
        decoder_cfg["d_model"] = dim
        decoder_cfg["d_ff"] = 4 * dim
        decoder = MaskTransformer(**decoder_cfg)
    else:
        raise ValueError(f"Unknown decoder: {name}")
    return decoder
