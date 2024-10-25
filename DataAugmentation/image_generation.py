import albumentations as A

# image first aug

def vertical_flip(image, p=1.0):
    transform = A.Compose([A.augmentations.VerticalFlip(p=p)])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image


def horizontal_flip(image, p=1.0):
    transform = A.Compose([A.augmentations.HorizontalFlip(p=p)])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image



def transpose(image, p=1.0):
    transform = A.Compose([A.augmentations.Transpose(p=p)])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image


def optical_distortion(image, distort_limit=1, shift_limit=0.5, interpolation=1, border_mode=4, value=None,
                      mask_value=None, always_apply=False, p=1.0):
    transform = A.Compose([A.augmentations.OpticalDistortion(distort_limit=distort_limit, shift_limit=shift_limit,
                                                             interpolation=interpolation, border_mode=border_mode,
                                                             value=value, mask_value=mask_value,
                                                             always_apply=always_apply, p=p)])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image


def grid_distortion(image, num_steps=5, distort_limit=0.3, interpolation=1, border_mode=4, value=None, mask_value=None,
                   always_apply=False, p=1):
    transform = A.Compose([A.augmentations.GridDistortion(num_steps=num_steps, distort_limit=distort_limit,
                                                          interpolation=interpolation, border_mode=border_mode,
                                                          value=value, mask_value=mask_value,
                                                          always_apply=always_apply, p=p)])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image