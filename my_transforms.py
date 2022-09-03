from albumentations import Compose, RandomBrightnessContrast, HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur, DualTransform
import cv2
import random
import numpy as np
import math
from scipy.ndimage import binary_dilation
import skimage.draw
from skimage import measure
from PIL import Image
random.seed(233)

def isotropically_resize_image(img, size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC):
    h, w = img.shape[:2]
    if max(w, h) == size:
        return img
    if w > h:
        scale = size / w
        h = h * scale
        w = size
    else:
        scale = size / h
        w = w * scale
        h = size
    h = size
    w = size
    interpolation = interpolation_up if scale > 1 else interpolation_down
    resized = cv2.resize(img, (int(w), int(h)), interpolation=interpolation)
    return resized


class IsotropicResize(DualTransform):
    def __init__(self, max_side, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC,
                 always_apply=False, p=1):
        super(IsotropicResize, self).__init__(always_apply, p)
        self.max_side = max_side
        self.interpolation_down = interpolation_down
        self.interpolation_up = interpolation_up

    def apply(self, img, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC, **params):
        return isotropically_resize_image(img, size=self.max_side, interpolation_down=interpolation_down,
                                          interpolation_up=interpolation_up)

    def apply_to_mask(self, img, **params):
        return self.apply(img, interpolation_down=cv2.INTER_NEAREST, interpolation_up=cv2.INTER_NEAREST, **params)

    def get_transform_init_args_names(self):
        return ("max_side", "interpolation_down", "interpolation_up")
    

def create_train_transforms(size=300):
    return Compose([
        ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
        GaussNoise(p=0.1),
        GaussianBlur(blur_limit=(3, 3), sigma_limit=(0, 0), p=0.05),
        HorizontalFlip(),
        OneOf([
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
        ], p=1),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.7),
        ToGray(p=0.2),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
    ]
    )

def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def split_eyes(image, landmarks):
    mask = np.zeros_like(image[..., 0])
    try:
        (x1, y1), (x2, y2) = landmarks[:2]
        mask = cv2.line(mask, (x1, y1), (x2, y2), color=(1), thickness=2)
        w = dist((x1, y1), (x2, y2))
        dilation = int(w // 4)
        mask = binary_dilation(mask, iterations=dilation)
    except Exception as e:
        pass
    return mask


def split_nose(image, landmarks):
    mask = np.zeros_like(image[..., 0])
    try:
        (x1, y1), (x2, y2) = landmarks[:2]
        x3, y3 = landmarks[2]
        x4 = int((x1 + x2) / 2)
        y4 = int((y1 + y2) / 2)
        mask = cv2.line(mask, (x3, y3), (x4, y4), color=(1), thickness=2)
        w = dist((x1, y1), (x2, y2))
        dilation = int(w // 4)
        mask = binary_dilation(mask, iterations=dilation)
    except Exception as e:
        pass
    return mask


def split_mouth(image, landmarks):
    mask = np.zeros_like(image[..., 0])
    try:
        (x1, y1), (x2, y2) = landmarks[-2:]
        mask = cv2.line(mask, (x1, y1), (x2, y2), color=(1), thickness=2)
        w = dist((x1, y1), (x2, y2))
        dilation = int(w // 3)
        mask = binary_dilation(mask, iterations=dilation)
    except Exception as e:
        pass
    return mask    

def region_erasing_1(img, landmarks):
    mask = np.ones_like(img[..., 0])
    mask_list = []
    mask_list.append(split_mouth(img, landmarks))
    mask_list.append(split_nose(img, landmarks))
    mask_list.append(split_eyes(img, landmarks))
    select_mask = random.sample(mask_list, random.randrange(1, len(mask_list) + 1))
    for _mask in select_mask:
        mask = mask * (1-_mask)
        
    img[mask==0] = 0
    
    return img, mask
    
    
    
    
def region_erasing_2(img, landmarks):
    mask = np.ones_like(img[..., 0])
    mask_list = []
    try:
        landmarks = np.array(landmarks)
        outline = landmarks[[*range(17), *range(26, 16, -1)]]
        Y, X = skimage.draw.polygon(outline[:, 1], outline[:, 0])
        _mask = np.zeros(img.shape[:2], dtype=np.uint8)
        _mask[Y, X] = 1
        y, x = measure.centroid(_mask)
        y = int(y)
        x = int(x)
        
        __mask = _mask.copy()
        __mask[:y, :] = 0
        __mask[:, :x] = 0
        mask_list.append(__mask)
        
        __mask = _mask.copy()
        __mask[:y, :] = 0
        __mask[:, x:] = 0
        mask_list.append(__mask)
        
        __mask = _mask.copy()
        __mask[y:, :] = 0
        __mask[:, :x] = 0
        mask_list.append(__mask)
        
        __mask = _mask.copy()
        __mask[y:, :] = 0
        __mask[:, x:] = 0
        mask_list.append(__mask)
    except Exception as e:
        pass
    
    select_mask = random.sample(mask_list, random.randrange(1, len(mask_list)))
    
    for _mask in select_mask:
        mask = mask * (1-_mask)
    
    img[mask==0] = 0

    return img, mask
    
if __name__ == '__main__':
    transforms = create_train_transforms(size = 224)
    print(transforms)