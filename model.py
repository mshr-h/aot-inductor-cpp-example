import os
import torch
from torchvision.models import get_model, get_model_weights
import torchvision


class Model(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super(Model, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).argmax(dim=1)


def main():
    model_name = "resnet18"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img_name = "cat.png"
    image_tensor = torchvision.io.decode_image(img_name)

    with torch.inference_mode():
        model = Model(get_model(model_name, weights="DEFAULT").eval()).to(device=device)
        weights = get_model_weights(model_name).DEFAULT
        transforms = weights.transforms()
        example_args = (transforms(image_tensor).unsqueeze(0).to(device),)
        ep_model = torch.export.export(model, example_args)
        output_path = torch._inductor.aoti_compile_and_package(
            ep_model,
            package_path=os.path.join(os.getcwd(), "model.pt2"),
        )


if __name__ == "__main__":
    main()
