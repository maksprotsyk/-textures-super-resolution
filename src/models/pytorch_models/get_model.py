from src.models.pytorch_models.UNet.unet_model import UNet
from src.models.pytorch_models.ResUNet.ResUNet import ResUNet
from src.models.pytorch_models.SwinIR.SwinIR import SwinIR


def get_model(model_name: str = 'SwinIR', **kwargs):
    print(f"{model_name=}")
    if model_name == 'UNet':
        return UNet(kwargs)
    elif model_name == 'ResUNet':
        print("fire")
        return ResUNet(num_res=20)
    elif model_name == 'SwinIR':
        return SwinIR(upscale=2, img_size=(128, 128),
                   window_size=8, img_range=1., depths=[6, 6, 6, 6],
                   embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect')
