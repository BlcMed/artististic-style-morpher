from torch import mm, nn


def calculate_content_loss(features_1, features_2, content_layer):
    white_noise_features_for_layer = features_1[content_layer]
    real_image_features_for_layer = features_2[content_layer]
    loss = nn.MSELoss()(white_noise_features_for_layer, real_image_features_for_layer)
    return loss


def _gram(features):
    a, b, c = features.shape
    t = features.view(a, b * c)
    gram = mm(t, t.t())
    return gram


def calculate_style_loss(features_1, features_2, style_layer):
    features_1_for_layer = features_1[style_layer]
    gram_1 = _gram(features_1_for_layer)

    features_2_for_layer = features_2[style_layer]
    gram_2 = _gram(features_2_for_layer)

    loss = nn.MSELoss()(gram_1, gram_2)
    return loss


def calculate_style_loss_for_layers(
    features_1, features_2, style_layers, style_layers_weights=None
):
    losses = []
    for layer in style_layers:
        loss = calculate_style_loss(features_1, features_2, style_layer=layer)
        losses.append(loss)
    total_style_loss = 0
    if style_layers_weights is None:
        total_style_loss = sum(losses) / len(losses)
    return total_style_loss


def calculate_mixed_loss(
    features_1,  # generated_features tipically from white noise
    content_features,
    style_features,
    content_layer,
    style_layers,
    content_weight,
    style_layers_weights=None,
):  # content_weight is alpha/beta, i.e. how much content contribute in image generation compared to style
    content_loss = calculate_content_loss(
        features_1=features_1, features_2=content_features, content_layer=content_layer
    )
    style_loss = calculate_style_loss_for_layers(
        features_1=features_1,
        features_2=style_features,
        style_layers=style_layers,
        style_layers_weights=style_layers_weights,
    )
    total_loss = content_loss * content_weight + style_loss
    return total_loss
