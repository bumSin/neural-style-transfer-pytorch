from torchvision import models

layerNameToIndexMap = {
    'relu1_2': 3,
    'relu2_2': 8,
    'relu3_3': 15,
    'relu4_3': 22
}
layerNames = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
layerIndexes = [3, 8, 15, 22]

activations = {}

# hook function to capture activations
def getActivations(layer_indx):
    def hook(model, input, output):
        activations[layer_indx] = output
    return hook

def getFeaturesOfStyleAndContentImage(model, style_image_tensor, content_image_tensor):
    model(content_image_tensor)
    content_image_features = [activations[layerIndex] for layerIndex in layerIndexes]

    model(style_image_tensor)
    style_image_features = [activations[layerIndex] for layerIndex in layerIndexes]

    return style_image_features, content_image_features[1]  # only relu2_2 is used for content features

def prepareModel(content_layer, style_layers):
    model = models.vgg16(pretrained=True)
    model.eval()

    # Freeze weights of all layers
    for param in model.parameters():
        param.requires_grad = False

    # add hooks
    for idx, (name, module) in enumerate(model.named_modules()):
        if idx in layerIndexes:
            module.register_forward_hook(getActivations(idx))

    return model

# makes a forward pass through models and returns the features accumulated by hooks
def forward_pass(model, batch_tensor):
    model(batch_tensor)
    feature_maps = [activations[layerIndex] for layerIndex in layerIndexes]  # removing the dummy dimension and putting all values in a list
    return feature_maps


# Layers to be exposed
# ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']

'''
VGG-16 architecture for reference:

VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)
'''