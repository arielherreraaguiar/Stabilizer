import PIL
import onnxruntime
import onnx
import io
import numpy as np
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch.onnx
import torch.nn as nn
import torch.nn.init as init
from matplotlib import pyplot as plt
import cv2 as cv
from unet import UNet
from utils.data_loading import BasicDataset
import argparse
from PIL import Image
from torchvision import transforms

if __name__ == "__main__":

    # Initialize model
    torch_model = UNet(n_channels=3, n_classes=2, bilinear=True)

    # Load pretrained model weights
    #batch_size =1    # just a random number

    # Initialize model with the pretrained weights
    torch_model.load_state_dict(torch.load("/home/inge/Pytorch-UNet/MODEL.pth"))

    # set the model to inference mode
    torch_model.eval()
    print(torch_model)

    # Input to the model
    full_img = Image.open("/home/inge/Pytorch-UNet/4.jpg")
    scale_factor=0.5
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = img.to(device=device, dtype=torch.float32)


    torch_out = torch_model(x)

    # Export the model
    torch.onnx.export(torch_model,               # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    "skyseg.onnx",   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=11,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'x' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})

    onnx_model = onnx.load("skyseg.onnx")
    onnx.checker.check_model(onnx_model)


    ort_session = onnxruntime.InferenceSession("skyseg.onnx")

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)
    

    #Output

    with torch.no_grad():
        output = torch_model(x)

        if torch_model.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()
    out_threshold = 0.5
    if torch_model.n_classes == 1:
        final_output = (full_mask > out_threshold).numpy()
    else:
        final_output = F.one_hot(full_mask.argmax(dim=0), torch_model.n_classes).permute(2, 0, 1).numpy()

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
    print(final_output.shape)
    #plt.imshow(final_output[1, :, :]+final_output[0, :, :])
    plt.imshow(final_output[1, :, :])
    plt.show()

