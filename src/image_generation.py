import torch
import torch.optim as optim

from .image_operations import generate_white_noise_image
from .loss import (
    calculate_content_loss,
    calculate_mixed_loss,
    calculate_style_loss_for_layers,
)
from .model_utils import get_features

WIDTH = 224
HEIGHT = 224


def reconstruct_image(
    model,
    transform,
    loss_function,
    num_iteration,
    learning_rate,
    generated_image_t=None,  # white noise image
    **loss_function_kwargs,  # depends on the type of reconstruction
):
    if generated_image_t is None:
        generated_image_t = generate_white_noise_image(
            width=WIDTH, height=HEIGHT, transform=transform
        )
    optimizer = optim.Adam([generated_image_t], learning_rate=learning_rate)

    for i in range(num_iteration):
        optimizer.zero_grad()

        generated_features = get_features(model, generated_image_t)
        loss = loss_function(
            features_1=generated_features,
            **loss_function_kwargs,
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


def reconstruct_image_from_style(
    style_image_t,
    style_layers,
    model,
    transform,
    num_iteration,
    learning_rate,
    style_layers_weights=None,
    generated_image_t=None,
):
    loss_function = calculate_style_loss_for_layers
    style_features = get_features(model, style_image_t)
    new_generated_image_t = reconstruct_image(
        model=model,
        transform=transform,
        loss_function=loss_function,
        num_iteration=num_iteration,
        learning_rate=learning_rate,
        generated_image_t=generated_image_t,
        features_2=style_features,
        style_layers=style_layers,
        style_layers_weights=style_layers_weights,
    )
    return new_generated_image_t


def reconstruct_image_from_content(
    content_image_t,
    content_layer,
    model,
    transform,
    num_iteration,
    learning_rate,
    generated_image_t=None,
):
    loss_function = calculate_content_loss
    content_features = get_features(model, content_image_t)
    new_generated_image_t = reconstruct_image(
        model=model,
        transform=transform,
        loss_function=loss_function,
        num_iteration=num_iteration,
        learning_rate=learning_rate,
        generated_image_t=generated_image_t,
        features_2=content_features,
        content_layer=content_layer,
    )
    return new_generated_image_t


def reconstruct_image_from_content_style(
    content_image_t,
    style_image_t,
    content_layer,
    style_layers,
    model,
    transform,
    num_iteration,
    learning_rate,
    content_weight,
    generated_image_t=None,
    style_layers_weights=None,
):
    loss_function = calculate_mixed_loss
    content_features = get_features(model, content_image_t)
    style_features = get_features(model, style_image_t)
    new_generated_image_t = reconstruct_image(
        model=model,
        transform=transform,
        loss_function=loss_function,
        num_iteration=num_iteration,
        learning_rate=learning_rate,
        generated_image_t=generated_image_t,
        content_features=content_features,
        style_features=style_features,
        content_layer=content_layer,
        style_layers=style_layers,
        style_layers_weights=style_layers_weights,
        content_weight=content_weight,
    )
    return new_generated_image_t


if __name__ == "__main__":
    from PIL import Image

    from .image_operations import image_to_tensor, tensor_to_image
    from .model_utils import load_model

    vgg_model, transform = load_model()

    # test style reconstruction
    style_image = Image.open("./data/starry_night.jpg")
    style_image_t = image_to_tensor(style_image, transform=transform)

    new_image_t = reconstruct_image_from_style(
        style_image_t=style_image_t,
        style_layers=[0],
        model=vgg_model,
        transform=transform,
        num_iteration=2,
        learning_rate=0.01,
    )
    new_image = tensor_to_image(new_image_t)

    # test content reconstruction
    content_image = Image.open("./data/dog.jpg")
    content_image_t = image_to_tensor(style_image, transform=transform)

    new_image_t = reconstruct_image_from_content(
        content_image_t=content_image_t,
        content_layer=0,
        model=vgg_model,
        transform=transform,
        num_iteration=2,
        learning_rate=0.01,
    )
    new_image = tensor_to_image(new_image_t)

    # test mixed reconstruction (content and style)
    new_image_t = reconstruct_image_from_content_style(
        content_image_t=content_image_t,
        style_image_t=style_image_t,
        content_layer=0,
        style_layers=list(range(4)),
        model=vgg_model,
        transform=transform,
        content_weight=0.1,
        num_iteration=1,
        learning_rate=0.01,
    )
    new_image = tensor_to_image(new_image_t)
    new_image.show()
