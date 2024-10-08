from torchvision.models.vgg import VGG19_Weights, vgg19


def load_model():
    weights = VGG19_Weights.DEFAULT
    model = vgg19(weights=weights)
    model.eval()
    transform = weights.transforms()
    return model, transform


def get_features(model, image_t):
    """
    forward pass and get the features for all layers
    """
    features = []
    features = {}
    for i, layer in enumerate(model.features):
        image_t = layer(image_t)
        features[i] = image_t
    return features
