import torch
import torch.optim as optim

from src.image_operations import generate_white_noise_image

from .loss import calculate_style_loss_for_layers
from .model_utils import get_features

WIDTH = 224
HEIGHT = 224


def reconstruct_image(
    # content_image_t,
    # style_image_t,
    # content_layer,
    # style_layers,
    # content_weight,
    model,
    transform,
    loss_function,
    num_iteration,
    lr,
    generated_image_t=None,  # white noise image
    **loss_function_kwargs,
):
    if generated_image_t is None:
        # _, width, height = content_image_features[0].shape
        generated_image_t = generate_white_noise_image(
            width=WIDTH, height=HEIGHT, transform=transform
        )
    optimizer = optim.Adam([generated_image_t], lr=lr)
    calculate_loss = loss_function

    for i in range(num_iteration):
        optimizer.zero_grad()

        generated_features = get_features(model, generated_image_t)
        # Calculate loss
        loss = calculate_loss(
            features_1=generated_features,
            **loss_function_kwargs,
            # content_features=content_image_features,
            # style_features=style_image_features,
            # content_layer=content_layer,
            # style_layers=style_layers,
            # content_weight=content_weight,
        )

        # Backward pass
        loss.backward(retain_graph=True)

        # Update the image
        optimizer.step()

        # Clip the values to be in the valid range
        with torch.no_grad():
            generated_image_t.data.clamp_(0, 1)

        print(f"Iteration {i+1}/{num_iteration}, Loss: {loss.item()}")
    return generated_image_t


def reconstruct_image_with_style(
    style_image_t,
    style_layers,
    model,
    transform,
    num_iteration,
    lr,
    weights=None,
    generated_image_t=None,
):
    loss_function = calculate_style_loss_for_layers
    style_features = get_features(model, style_image_t)
    new_generated_image_t = reconstruct_image(
        model=model,
        transform=transform,
        loss_function=loss_function,
        num_iteration=num_iteration,
        lr=lr,
        generated_image_t=generated_image_t,
        features_2=style_features,
        style_layers=style_layers,
        weights=weights,
    )
    return new_generated_image_t


if __name__ == "__main__":
    from PIL import Image

    from .image_operations import image_to_tensor, tensor_to_image
    from .model_utils import load_model

    model, transform = load_model()
    style_image = Image.open("./data/starry_night.jpg")
    style_image_t = image_to_tensor(style_image, transform=transform)

    new_image_t = reconstruct_image_with_style(
        style_image_t=style_image_t,
        style_layers=[0],
        model=model,
        transform=transform,
        num_iteration=10,
        lr=0.01,
    )
    new_image = tensor_to_image(new_image_t)
    new_image.show()
