import albumentations as A
import cv2
import random

class ImageAugmentation:
    def __init__(self):
        self.augmentations = [
            self.blur, self.clahe, self.channel_dropout, self.downscale, self.emboss,
            self.gauss_noise, self.gaussian_blur, self.glass_blur,
            self.hue_saturation_value, self.iso_noise, self.invert_img, self.median_blur,
            self.motion_blur, self.multiplicative_noise, self.poster_ize, self.rgb_shift,
            self.sharpen, self.solarize, self.superpixels, self.to_gray, self.to_sepia,
            self.jpeg_compression, self.coarse_dropout
        ]

    def random_aug(self, image):
        augmentation = random.choice(self.augmentations)
        return augmentation(image)

    @staticmethod
    # all use
    def apply_transform(transform, image):
        transformed = transform(image=image)
        return transformed["image"]

    @staticmethod
    def blur(image, blur_limit=(3, 7), always_apply=False, p=1.0):
        transform = A.Compose([A.Blur(blur_limit=blur_limit, always_apply=always_apply, p=p)])
        return ImageAugmentation.apply_transform(transform, image)

    @staticmethod
    def clahe(image, clip_limit=(1, 4), tile_grid_size=(8, 8), always_apply=False, p=1.0):
        transform = A.Compose([A.CLAHE(clip_limit=clip_limit, tile_grid_size=tile_grid_size, always_apply=always_apply, p=p)])
        return ImageAugmentation.apply_transform(transform, image)

    @staticmethod
    def channel_dropout(image, channel_drop_range=(1, 1), fill_value=0, always_apply=False, p=1.0):
        transform = A.Compose([A.ChannelDropout(channel_drop_range=channel_drop_range, fill_value=fill_value, always_apply=always_apply, p=p)])
        return ImageAugmentation.apply_transform(transform, image)

    @staticmethod
    def downscale(image, scale_min=0.25, scale_max=0.25, interpolation=0, always_apply=False, p=1.0):
        transform = A.Compose([A.Downscale(scale_min=scale_min, scale_max=scale_max, interpolation=interpolation, always_apply=always_apply, p=p)])
        return ImageAugmentation.apply_transform(transform, image)

    @staticmethod
    def emboss(image, alpha=(0.2, 0.5), strength=(2, 5), always_apply=False, p=1.0):
        transform = A.Compose([A.Emboss(alpha=alpha, strength=strength, always_apply=always_apply, p=p)])
        return ImageAugmentation.apply_transform(transform, image)

    @staticmethod
    def gauss_noise(image, var_limit=(30.0, 50.0), mean=25, per_channel=True, always_apply=False, p=1.0):
        transform = A.Compose([A.GaussNoise(var_limit=var_limit, mean=mean, per_channel=per_channel, always_apply=always_apply, p=p)])
        return ImageAugmentation.apply_transform(transform, image)

    @staticmethod
    def gaussian_blur(image, blur_limit=(5, 7), sigma_limit=10, always_apply=False, p=1.0):
        transform = A.Compose([A.GaussianBlur(blur_limit=blur_limit, sigma_limit=sigma_limit, always_apply=always_apply, p=p)])
        return ImageAugmentation.apply_transform(transform, image)

    @staticmethod
    def glass_blur(image, sigma=0.7, max_delta=4, iterations=2, always_apply=False, p=1.0):
        transform = A.Compose([A.GlassBlur(sigma=sigma, max_delta=max_delta, iterations=iterations, always_apply=always_apply, mode='fast', p=p)])
        return ImageAugmentation.apply_transform(transform, image)

    @staticmethod
    def hue_saturation_value(image, hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=1.0):
        transform = A.Compose([A.HueSaturationValue(hue_shift_limit=hue_shift_limit, sat_shift_limit=sat_shift_limit, val_shift_limit=val_shift_limit, always_apply=always_apply, p=p)])
        return ImageAugmentation.apply_transform(transform, image)

    @staticmethod
    def iso_noise(image, color_shift=(0.1, 0.5), intensity=(0.1, 0.5), always_apply=False, p=1.0):
        transform = A.Compose([A.ISONoise(color_shift=color_shift, intensity=intensity, always_apply=always_apply, p=p)])
        return ImageAugmentation.apply_transform(transform, image)

    @staticmethod
    def invert_img(image, p=1.0):
        transform = A.Compose([A.InvertImg(p=p)])
        return ImageAugmentation.apply_transform(transform, image)

    @staticmethod
    def median_blur(image, blur_limit=7, always_apply=False, p=1.0):
        transform = A.Compose([A.MedianBlur(blur_limit=blur_limit, always_apply=always_apply, p=p)])
        return ImageAugmentation.apply_transform(transform, image)

    @staticmethod
    def motion_blur(image, blur_limit=7, p=1.0):
        transform = A.Compose([A.MotionBlur(blur_limit=blur_limit, p=p)])
        return ImageAugmentation.apply_transform(transform, image)

    @staticmethod
    def multiplicative_noise(image, multiplier=(0.8, 1.2), per_channel=False, p=1.0):
        transform = A.Compose([A.MultiplicativeNoise(multiplier=multiplier, per_channel=per_channel, p=p)])
        return ImageAugmentation.apply_transform(transform, image)

    @staticmethod
    def poster_ize(image, num_bits=4, always_apply=False, p=1.0):
        transform = A.Compose([A.Posterize(num_bits=num_bits, always_apply=always_apply, p=p)])
        return ImageAugmentation.apply_transform(transform, image)

    @staticmethod
    def rgb_shift(image, r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, always_apply=False, p=1.0):
        transform = A.Compose([A.RGBShift(r_shift_limit=r_shift_limit, g_shift_limit=g_shift_limit, b_shift_limit=b_shift_limit, always_apply=always_apply, p=p)])
        return ImageAugmentation.apply_transform(transform, image)

    @staticmethod
    def sharpen(image, alpha=(0.2, 0.5), lightness=(0.5, 1.0), always_apply=False, p=1.0):
        transform = A.Compose([A.Sharpen(alpha=alpha, lightness=lightness, always_apply=always_apply, p=p)])
        return ImageAugmentation.apply_transform(transform, image)

    @staticmethod
    def solarize(image, threshold=128, always_apply=False, p=1.0):
        transform = A.Compose([A.Solarize(threshold=threshold, always_apply=always_apply, p=p)])
        return ImageAugmentation.apply_transform(transform, image)

    @staticmethod
    def superpixels(image, p_replace=0.3, n_segments=100, max_size=128, interpolation=1, always_apply=False, p=1.0):
        transform = A.Compose([A.Superpixels(p_replace=p_replace, n_segments=n_segments, max_size=max_size, interpolation=interpolation, always_apply=always_apply, p=p)])
        return ImageAugmentation.apply_transform(transform, image)

    @staticmethod
    def to_gray(image, p=1.0):
        transform = A.Compose([A.ToGray(p=p)])
        return ImageAugmentation.apply_transform(transform, image)

    @staticmethod
    def to_sepia(image, always_apply=False, p=1.0):
        transform = A.Compose([A.ToSepia(always_apply=always_apply, p=p)])
        return ImageAugmentation.apply_transform(transform, image)

    @staticmethod
    def jpeg_compression(image, quality=95):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, compressed_image = cv2.imencode('.jpg', image, encode_param)
        transformed_image = cv2.imdecode(compressed_image, 1)
        return transformed_image

    @staticmethod
    def coarse_dropout(image, max_holes=8, max_height=8, max_width=8, min_holes=None, min_height=None, min_width=None, fill_value=0, mask_fill_value=None, always_apply=False, p=1.0):
        transform = A.Compose([A.CoarseDropout(max_holes=max_holes, max_height=max_height, max_width=max_width, min_holes=min_holes, min_height=min_height, min_width=min_width, fill_value=fill_value, mask_fill_value=mask_fill_value, always_apply=always_apply, p=p)])
        return ImageAugmentation.apply_transform(transform, image)

