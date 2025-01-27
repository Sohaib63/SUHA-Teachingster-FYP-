from io import BytesIO

import numpy as np
from PIL import Image
input_shape=(224,224)
def read_image(image):
    plt_image=Image.open(BytesIO(image))
    return plt_image
def preprocess_image(image):
    image=image.resize(input_shape)
    image=np.asfarray(image)
    image=image/127.5-1.0
    image=np.expand_dims(image, 0)
    return image