import argparse
import os.path


def initiate_style_transfer(arguments):
    pass


if __name__ == "__main__":
    default_resource_dir = os.path.join(os.path.dirname(__file__), 'data')
    content_images_dir = os.path.join(default_resource_dir, 'content_image')
    style_images_dir = os.path.join(default_resource_dir, 'style_image')
    output_images_dir = os.path.join(default_resource_dir, 'style_transferred_image')
    img_format = (4, '.jpg')  # saves images in the format: %04d.jpg

    parser = argparse.ArgumentParser()
    parser.add_argument("--content_img_name", type=str, help="content image file name", default="")
    parser.add_argument("--style_img_name", type=str, help="style image file name", default="")

    args = parser.parse_args()

    # putting all the args and dir names in a dict
    arguments = dict()

    for arg in vars(args):
        arguments[arg] = getattr(args, arg)

    arguments['content_images_dir'] = content_images_dir
    arguments['style_images_dir'] = style_images_dir
    arguments['output_images_dir'] = output_images_dir

    initiate_style_transfer(arguments)