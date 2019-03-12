import PIL
import torchvision.transforms as transforms
from torch.autograd import Variable


def prepare_image(image):
    image = preprocess_image(image)
    im_var = convert_img_var(image)
    return im_var


def load_image(filepath):
    return PIL.Image.open(filepath).convert('RGB')


def preprocess_image(image):
    # Create data transforms
    TARGET_IMAGE_SIZE = [448, 448]
    CHANNEL_MEAN = [0.485, 0.456, 0.406]
    CHANNEL_STD = [0.229, 0.224, 0.225]
    data_transforms = transforms.Compose([transforms.Resize(TARGET_IMAGE_SIZE),
                                          transforms.ToTensor(),
                                          transforms.Normalize(CHANNEL_MEAN,
                                                               CHANNEL_STD)])
    # apply data transforms
    img_transform = data_transforms(image)
    # make sure grey scale image is processed correctly
    if img_transform.shape[0] == 1:
        img_transform = img_transform.expand(3, -1, -1)
    img_transform = img_transform.unsqueeze(0)

    return img_transform


def convert_img_var(image):
    return Variable(image).cuda()
