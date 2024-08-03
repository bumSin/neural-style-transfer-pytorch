import torch


def getImageTensor(param, style_img_name):
    return torch.rand(3, 224, 224)


def convertImageToVGG16InputTensor(image_path):
    img = Image.open(image_path)
    img = img.convert("RGB")  # remove alpha

    # preprocessing steps
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])  # Normalize with the VGG16 mean and std
    ])

    # apply the preprocessing to image
    image_tensor = preprocess(img)
    return image_tensor

def save_as_Image(init_image_tensor, dir1):
    pass