from torch import nn, mm


def calculate_content_loss(features_1, features_2, layer):
    white_noise_features_for_layer = features_1[layer]
    real_image_features_for_layer = features_2[layer]
    loss = nn.MSELoss()(white_noise_features_for_layer, real_image_features_for_layer)
    return loss


def _gram(features):
    a, b, c = features.shape
    t = features.view(a, b * c)
    gram = mm(t, t.t())
    return gram


def calculate_style_loss(features_1, features_2, layer):
    features_1_for_layer = features_1[layer]
    gram_1 = _gram(features_1_for_layer)

    features_2_for_layer = features_2[layer]
    gram_2 = _gram(features_2_for_layer)

    loss = nn.MSELoss()(gram_1, gram_2)
    return loss


def calculate_style_loss_for_layers(features_1, features_2, layers, weights=None):
    losses = []
    for layer in layers:
        loss = calculate_style_loss(features_1, features_2, layer=layer)
        losses.append(loss)
    total_style_loss = 0
    if weights is None:
        total_style_loss = sum(losses) / len(losses)
    return total_style_loss


def calculate_total_loss(
    content_features,
    style_features,
    generated_features,
    content_layer,
    style_layers,
    content_weight,
    layers_weights=None,
):  # content_weight is alpha/beta, i.e. how much content contribute in image generation compared to style
    content_loss = calculate_content_loss(
        features_1=content_features, features_2=generated_features, layer=content_layer
    )
    style_loss = calculate_style_loss_for_layers(
        features_1=style_features,
        features_2=generated_features,
        layers=style_layers,
        weights=layers_weights,
    )
    total_loss = content_loss * content_weight + style_loss
    return total_loss
