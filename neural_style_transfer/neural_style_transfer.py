import torch
import argparse
import os.path

from utils.ImageUtils import *
from utils.ModelUtils import *
from services.LossService import *



def initiate_style_transfer(arguments):
    content_image_tensor = getImageTensor(arguments['content_images_dir'], arguments['content_img_name'])
    style_image_tensor = getImageTensor(arguments['style_images_dir'], arguments['style_img_name'])

    # visualize_tensor(content_image_tensor)
    # visualize_tensor(style_image_tensor)

    if arguments['init_strategy'] == 'white_noise':
        init_image_tensor = torch.rand(3, 224, 224)
    else:
        init_image_tensor = content_image_tensor

    # We will backprop on the pixels of this image, hence we need to track gradients
    init_image_tensor.requires_grad_()

    # Defining layers to be used for content regeneration and style regenerations
    # I am using what was used in original paper used but this is configurable
    content_layer = 'relu2_2'
    style_layers = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']

    model = prepareModel(content_layer, style_layers)
    optimizer = torch.optim.Adam([init_image_tensor], lr=0.01)

    # Features of style image and content image won't change, hence calculating only once
    style_image_features, content_image_features = getFeaturesOfStyleAndContentImage(model,
        style_image_tensor, content_image_tensor)

    gram_matrix_style_features = [gram_matrix(feature) for feature in style_image_features]

    # for feature, gram_feature in zip(style_image_features, gram_matrix_style_features):
    #     print(f"Shape of style_image_features: {feature.shape} and shape of "
    #           f"content_features_of_content_image: {content_image_features.shape} "
    #           f"gram_matrix_style_features : {gram_feature.shape}")

    for i in range(arguments['epochs']):
        optimizer.zero_grad()

        init_image_features = forward_pass(init_image_tensor)
        loss = calculateLoss(init_image_features, style_image_features, content_image_features, gram_matrix_style_features)

        loss.backward()
        optimizer.step()

    # save_as_Image(init_image_tensor, arguments.output_images_dir)


if __name__ == "__main__":
    default_resource_dir = os.path.join(os.path.dirname(__file__), 'data')
    content_images_dir = os.path.join(default_resource_dir, 'content_image')
    style_images_dir = os.path.join(default_resource_dir, 'style_image')
    output_images_dir = os.path.join(default_resource_dir, 'style_transferred_image')
    img_format = (4, '.jpg')  # saves images in the format: %04d.jpg

    parser = argparse.ArgumentParser()
    parser.add_argument("--content_img_name", type=str, help="content image file name", default="cute_doggo.png")
    parser.add_argument("--style_img_name", type=str, help="style image file name", default="starry_night.png")
    parser.add_argument("--init_strategy", type=str, help="strategy to initiate base image", default="content")
    parser.add_argument("--epochs", type=int, help="count of epochs", default=1)
    parser.add_argument("--alpha", type=float, help="weight factor for content loss", default=1e5)
    parser.add_argument("--beta", type=float, help="weight factor for style loss", default=3e4)

    args = parser.parse_args()

    # putting all the args and dir names in a dict
    arguments = dict()

    for arg in vars(args):
        arguments[arg] = getattr(args, arg)

    arguments['content_images_dir'] = content_images_dir
    arguments['style_images_dir'] = style_images_dir
    arguments['output_images_dir'] = output_images_dir

    initiate_style_transfer(arguments)