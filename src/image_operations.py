from torch import randn
from torchvision import transforms
from PIL import Image


def read_image(image_path):
    image = Image.open(image_path)
    return image


def image_to_tensor(image, transform, requires_grad=False):
    image_t = transform(image)
    image_t.requires_grad = requires_grad
    return image_t


def tensor_to_image(image_t):
    image = transforms.Compose([transforms.ToPILImage()])(
        image_t.squeeze(0).cpu().clone().detach()
    )
    return image


def generate_white_noise_image(width, height, transform):
    white_noise_image = randn(3, width, height)
    white_noise_image = transform(white_noise_image)
    white_noise_image.requires_grad = True
    return white_noise_image
