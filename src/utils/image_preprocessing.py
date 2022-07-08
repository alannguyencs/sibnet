from constants import *
from skimage import transform, io

INPUT_SIZE = (224,224)

def preprocessing_image(image_path):
    image = np.asarray(Image.open(image_path).convert('RGB').resize(INPUT_SIZE, Image.ANTIALIAS)) / 255
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image).float()
    for t, m, s in zip(image, IMAGENET_MEAN, IMAGENET_STD):
        t.sub_(m).div_(s)
    image = torch.unsqueeze(image, 0)
    return image