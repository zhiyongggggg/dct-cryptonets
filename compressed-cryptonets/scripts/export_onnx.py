import sys, os
import torch
import onnx
from torchinfo import summary

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from models.model import Model
from models.pkn import PKN2d, convwrapper


if __name__ == "__main__":
    model = Model(model_name='resnet10', num_classes=10, pretrained=True).to('cuda')
    convwrapper(model, PKN2d, device='cuda')
    print(model)
    # summary(model, (3, 224, 224))

    checkpoint = torch.load('resnet10_model.pth', map_location='cuda')
    state_keys = list(checkpoint.keys())
    print(state_keys)
    model_dict_load = model.state_dict()
    model_dict_load.update(checkpoint)
    model.load_state_dict(model_dict_load)

    print('Exporting ONNX...')
    torch.onnx.export(
        model,
        torch.randn(1, 3, 224, 224).to('cuda'),
        'polykernet2.onnx',
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        verbose=False,
    )
    onnx_model = onnx.load("polykernet2.onnx")
    onnx.checker.check_model(onnx_model)
