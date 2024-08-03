import torch

def calculateLoss(init_image_features, content_image_features, arguments, gram_matrix_style_features):
    style_loss = calculateStyleLoss(init_image_features, gram_matrix_style_features)
    content_loss = torch.nn.MSELoss(reduction='mean')(init_image_features, content_image_features)  # input, target

    return arguments['alpha']*content_loss + arguments['beta']*style_loss

def calculateStyleLoss(init_image_features, gram_matrix_style_features):
    init_image_gram_matrix = [gram_matrix(feature) for feature in init_image_features]

    style_loss = 0.0

    for init_gram, style_gram in zip(init_image_gram_matrix, gram_matrix_style_features):
        style_loss += torch.nn.MSELoss(reduction='sum')(init_gram, style_gram)

    style_loss /= len(init_image_gram_matrix)

def gram_matrix(input_tensor):
    # Get the dimensions of the input tensor
    channels, height, width = input_tensor.size()

    # Reshape the tensor to (C, H*W)
    features = input_tensor.view(channels, height * width)

    # Compute the Gram matrix
    G = torch.mm(features, features.t())

    # Normalize the Gram matrix
    G = G.div(channels * height * width)

    return G