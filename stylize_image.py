import torch
from models import loss_models, transformation_models
from torchvision.transforms.functional import pil_to_tensor, resize
from utils import preprocess_image, deprocess_image


def stylize_image(image, image_size, model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mean = loss_models.VGG16Loss.MEAN.to(device)
    std = loss_models.VGG16Loss.STD.to(device)

    img = pil_to_tensor(image.convert("RGB")).to(device)
    if image_size:
        img = resize(img, size=image_size)
    img = preprocess_image(img, mean, std)

    transformation_model = (
        transformation_models.TransformationModel().to(device).eval()
    )

    # code to load pretrained model
    checkpoint = torch.load(model_path, map_location=device)
    transformation_model.load_state_dict(checkpoint["model_state_dict"])

    transformation_model.requires_grad_(False)

    gen_image = transformation_model(img)
    gen_image = deprocess_image(gen_image, mean, std)

    return gen_image.squeeze(0)
