from constants import *
from skimage import transform, io

INPUT_SIZE = (224,224)

def preprocessing_image(image_path):
    image = np.asarray(Image.open(image_path).convert('RGB').resize(INPUT_SIZE, Image.ANTIALIAS)) / 255
    # np_image = io.imread(image_path)
    # image = transform.resize(np_image, INPUT_SIZE)
    # image = np.moveaxis(image, 2, 0)
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image).float()
    # image = image.transpose(1, 2).transpose(0, 1)
    for t, m, s in zip(image, IMAGENET_MEAN, IMAGENET_STD):
        t.sub_(m).div_(s)
    image = torch.unsqueeze(image, 0)
    return image