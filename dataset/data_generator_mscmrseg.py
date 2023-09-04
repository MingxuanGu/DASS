import os.path
import re
from pathlib import Path
import math

import imgaug
import numpy as np
from torch.utils import data
import cv2
from glob import glob
import imgaug.augmenters as iaa
import elasticdeform

import config
from utils.utils_ import tranfer_data_2_scratch


def to_categorical(mask, num_classes, channel='channel_first'):
    """
    Convert label into categorical format (one-hot encoded)
    Args:
        mask: The label to be converted
        num_classes: maximum number of classes in the label
        channel: whether the output mask should be 'channel_first' or 'channel_last'

    Returns:
    The one-hot encoded label
    """
    if channel != 'channel_first' and channel != 'channel_last':
        assert False, r"channel should be either 'channel_first' or 'channel_last'"
    assert num_classes > 1, "num_classes should be greater than 1"
    unique = np.unique(mask)
    assert len(unique) <= num_classes, "number of unique values should be smaller or equal to the num_classes"
    assert np.max(unique) < num_classes, "maximum value in the mask should be smaller than the num_classes"
    if mask.shape[1] == 1:
        mask = np.squeeze(mask, axis=1)
    if mask.shape[-1] == 1:
        mask = np.squeeze(mask, axis=-1)
    eye = np.eye(num_classes, dtype='uint8')
    output = eye[mask]
    if channel == 'channel_first':
        output = np.moveaxis(output, -1, -3)
    return output


class ImageProcessor:
    @staticmethod
    def aug(image, mask):
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        seq = iaa.Sequential(
            [
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                sometimes(iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    # scale images to 80-120% of their size, individually per axis
                    translate_percent={"x": (-0.1, 0.05), "y": (-0.1, 0.1)},
                    rotate=(-10, 10),  # rotate by -10 to +10 degrees
                    shear=(-12, 12),  # shear by -12 to +12 degrees
                    order=[0, 1],
                    cval=(0, 255),
                    mode='constant'
                )),
            ],
            random_order=True
        )
        if image.ndim == 4:
            mask = np.array(mask)
            image_heavy, mask_heavy = seq(images=image, segmentation_maps=mask.astype(np.int32))
        else:
            image_heavy, mask_heavy = seq(images=image[np.newaxis, ...], segmentation_maps=mask[np.newaxis, ...])
            image_heavy, mask_heavy = image_heavy[0], mask_heavy[0]
        return image_heavy, mask_heavy

    @staticmethod
    def simple_aug(image, mask):
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        seq = iaa.Sequential(
            [
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                sometimes(iaa.geometric.Rot90(k=imgaug.ALL)),
            ],
            random_order=True
        )
        if image.ndim == 4:
            mask = np.array(mask)
            image_heavy, mask_heavy = seq(images=image, segmentation_maps=mask.astype(np.int32))
        else:
            image_heavy, mask_heavy = seq(images=image[np.newaxis, ...], segmentation_maps=mask[np.newaxis, ...])
            image_heavy, mask_heavy = image_heavy[0], mask_heavy[0]
        return image_heavy, mask_heavy

    @staticmethod
    def heavy_aug(image, mask, vmax=255):
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        seq = iaa.Sequential(
            [
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                sometimes(iaa.geometric.Rot90(k=imgaug.ALL)),
                sometimes(iaa.SomeOf((0, 3),
               [
                   iaa.OneOf([
                       iaa.GaussianBlur((0, 2.0)),
                       iaa.AverageBlur(k=(2, 6)),
                       iaa.MedianBlur(k=(3, 5)),
                   ]),
                   iaa.AdditiveGaussianNoise(
                       loc=0, scale=(0.0, 0.1 * vmax), per_channel=0.5
                   ),
                   iaa.OneOf([
                       iaa.Dropout((0.01, 0.1), per_channel=0.5),
                       iaa.CoarseDropout(
                           (0.01, 0.05), size_percent=(0.04, 0.1),
                           per_channel=0.2
                       ),
                   ]),
                   # sometimes(
                   #     iaa.Superpixels(
                   #         p_replace=(0, 1.0),
                   #         n_segments=(20, 200)
                   #     )
                   # ),
                   # iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                   # iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
                   # sometimes(iaa.OneOf([
                   #     iaa.EdgeDetect(alpha=(0, 0.7)),
                   #     iaa.DirectedEdgeDetect(
                   #         alpha=(0, 0.7), direction=(0.0, 1.0)
                   #     ),
                   # ])),
                   # iaa.Invert(0.05, per_channel=True),
                   # iaa.Add((-10, 10), per_channel=0.5),
                   # iaa.Multiply((0.5, 1.5), per_channel=0.5),
                   # iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
                   # iaa.Grayscale(alpha=(0.0, 1.0)),
                   # # sometimes(
                   # #     iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                   # # ),
                   # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
               ]),)
            ],
            random_order=True
        )
        if image.ndim == 4:
            mask = np.array(mask)
            image_heavy, mask_heavy = seq(images=image, segmentation_maps=mask.astype(np.int32))
        else:
            image_heavy, mask_heavy = seq(images=image[np.newaxis, ...], segmentation_maps=mask[np.newaxis, ...].astype(np.int32))
            image_heavy, mask_heavy = image_heavy[0], mask_heavy[0]
        return image_heavy, mask_heavy

    @staticmethod
    def crop_volume(vol, crop_size=112):
        """
        Center crop the input vol into [B, 2 * crop_size, 2 * crop_size, ...]
        :param vol:
        :param crop_size:
        :return:
        """
        if vol.ndim == 3 or vol.ndim == 2:
            return np.array(vol[int(vol.shape[0] / 2) - crop_size: int(vol.shape[0] / 2) + crop_size,
                            int(vol.shape[1] / 2) - crop_size: int(vol.shape[1] / 2) + crop_size, ])
        elif vol.ndim == 4:
            return np.array(vol[:, int(vol.shape[1] / 2) - crop_size: int(vol.shape[1] / 2) + crop_size,
                            int(vol.shape[2] / 2) - crop_size: int(vol.shape[2] / 2) + crop_size, ])
        else:
            raise ValueError(f'the number of dimension of the image should be among (2, 3, 4), {vol.ndim} detected.')


class DataGenerator(data.Dataset):
    def __init__(self, phase="train", modality="bssfp", crop_size=224, n_samples=-1, augmentation=False, clahe=False,
                 data_dir='../data/mscmrseg/origin', pat_id=-1, slc_id=-1, bs=16, aug_mode='simple', aug_counter=False,
                 normalization='minmax', fold=0, domain='s', vert=False):
        assert modality == "bssfp" or modality == "t2" or modality == 'lge'
        self._modality = modality
        self._crop_size = crop_size
        self._phase = phase
        self._index = 0  # start from the 0th sample
        self._totalcount = 0
        self._augmentation = augmentation
        self._aug_mode = aug_mode
        self._aug_counter = aug_counter
        self._normalization = normalization
        self._vert = vert
        # if normalization == 'zscore':
        #     self._df = pd.read_csv(Path(data_dir).joinpath('mscmrseg_uint8_mean_std.csv'))
        self._ifclahe = clahe
        if clahe:
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        if modality == 'bssfp':
            pat_name = 'bSSFP'
        else:
            pat_name = modality
        st = 'A' if (modality == 'bssfp' or modality == 't2') else 'B'
        # pat_id = '*' if pat_id == -1 else pat_id
        # slc_id = '*' if slc_id == -1 else slc_id
        self._image_files, self._mask_files, self._vert_files = [], [], []
        if domain == 't':
            pat_ids = config.MSCMRSEG_TEST_FOLD1 if fold == 0 else config.MSCMRSEG_TEST_FOLD2
        elif domain == 's':
            pat_ids = config.MSCMRSEG_TEST_FOLD1 + config.MSCMRSEG_TEST_FOLD2
        else:
            raise NotImplementedError
        for pat_id in pat_ids:
            self._image_files += glob(os.path.join(data_dir, f'train{st}/pat_{pat_id}_{pat_name}_{"*"}.png'))
            self._mask_files += glob(os.path.join(data_dir, f'train{st}mask/pat_{pat_id}_{pat_name}_{"*"}.png'))
            if vert:
                self._vert_files += glob(os.path.join(data_dir, f'vert{st}/pat_{pat_id}_{pat_name}_{"*"}.npy'))
        assert len(self._image_files) == len(self._mask_files) and \
               len(self._image_files) > 0, f'data dir: {data_dir}, img file len: {len(self._image_files)}, ' \
                                           f'mask file len: {len(self._mask_files)}'
        self._len = len(self._image_files)
        print("{}: {}".format(modality, self._len))
        # self._shuffle_indices = np.arange(self._len)
        # self._shuffle_indices = np.random.permutation(self._shuffle_indices)
        if n_samples == -1:
            self._n_samples = self._len + self._len % bs
        else:
            self._n_samples = n_samples
        self._names = [Path(file).stem.split('.')[0] for file in self._image_files]

    def __len__(self):
        return self._n_samples

    @property
    def n_samples(self):
        return self._n_samples

    @n_samples.setter
    def n_samples(self, value):
        self._n_samples = value

    def get_images_masks(self, img_path, mask_path):
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.where(mask == 85, 1, mask)
        mask = np.where(mask == 212, 2, mask)
        mask = np.where(mask == 255, 3, mask)
        return img, mask

    def __getitem__(self, index):

        i = index % self._len
        img, mask = self.get_images_masks(img_path=self._image_files[i], mask_path=self._mask_files[i])
        if self._ifclahe:
            img = self.clahe.apply(img[..., 0])
            img = np.stack([img, img, img], axis=-1)
        aug_img, aug_mask = img, mask
        if self._augmentation:
            aug_mask = np.expand_dims(aug_mask, axis=-1)
            if self._aug_mode == 'simple':
                aug_img, aug_mask = ImageProcessor.simple_aug(image=aug_img, mask=aug_mask)
            else:
                aug_img, aug_mask = ImageProcessor.heavy_aug(image=aug_img, mask=aug_mask)
                if np.random.uniform(0, 1) < .5:
                    [aug_img, aug_mask] = elasticdeform.deform_random_grid([aug_img, aug_mask], axis=[(0, 1), (0, 1)],
                                                                           sigma=np.random.uniform(1, 7), order=0,
                                                                           mode='constant')
            aug_mask = aug_mask[..., 0]
        # elif self._normalization == 'zscore':
        #     pat_id = int(re.search(r'_\d+_', self._names[i]).group()[1:-1])
        #     entry = self._df[(self._df['modality'] == self._modality) & (self._df['pat_id'] == pat_id)]
        #     mean = entry['mean'].values[0]
        #     std = entry['std'].values[0]
        #     aug_img = (np.array(aug_img, np.float32) - mean) / std
        if self._crop_size and aug_img.shape[1] != self._crop_size:
            aug_img = ImageProcessor.crop_volume(aug_img, crop_size=self._crop_size // 2)
            aug_mask = ImageProcessor.crop_volume(np.array(aug_mask), crop_size=self._crop_size // 2)
        aug_img = np.moveaxis(aug_img, -1, -3)
        if self._normalization == 'minmax':
            aug_img = np.array(aug_img, np.float32) / 255.
        elif self._normalization == 'zscore':
            aug_img = (np.array(aug_img, np.float32) - aug_img.mean()) / aug_img.std()
        else:
            raise NotImplementedError
        if self._vert:
            vertices = np.load(self._vert_files[i])  # (300, 3)
            return aug_img, aug_mask, vertices
        if self._aug_counter:
            if self._augmentation:
                mask = np.expand_dims(mask, axis=-1)
                if self._aug_mode == 'simple':
                    img, _ = ImageProcessor.simple_aug(image=img, mask=mask)
                else:
                    img, _ = ImageProcessor.heavy_aug(image=img, mask=mask)
                # mask = mask[..., 0]
            if self._crop_size and img.shape[1] != self._crop_size:
                img = ImageProcessor.crop_volume(img, crop_size=self._crop_size // 2)
            img = np.moveaxis(img, -1, -3)
            # mask = to_categorical(np.array(mask), num_classes=4)
            if self._normalization == 'minmax':
                img = np.array(img, np.float32) / 255.
            elif self._normalization == 'zscore':
                img = (np.array(img, dtype=np.float32) - img.mean()) / img.std()
            else:
                raise NotImplementedError
            return aug_img, img, self._names[i]  # (3, 256, 256) (4, 256, 256)
        else:
            return aug_img, aug_mask, self._names[i]


def prepare_dataset(args, aug_counter=False, vert=False):
    scratch = tranfer_data_2_scratch(args.data_dir, args.scratch)
    scratch_raw = tranfer_data_2_scratch(args.raw_data_dir, args.scratch)
    content_dataset = DataGenerator(modality='lge' if args.rev else 'bssfp', crop_size=args.crop,
                                    augmentation=args.aug_s, data_dir=scratch, bs=args.bs, clahe=args.clahe,
                                    aug_mode=args.aug_mode, normalization=args.normalization, fold=args.fold,
                                    aug_counter=aug_counter if args.rev else False, domain='s', vert=vert)
    style_dataset = DataGenerator(modality='bssfp' if args.rev else 'lge', crop_size=args.crop,
                                  augmentation=args.aug_t, data_dir=scratch, bs=args.bs, clahe=args.clahe,
                                  aug_mode=args.aug_mode, normalization=args.normalization, fold=args.fold,
                                  aug_counter=False if args.rev else aug_counter, domain='t', vert=vert)
    n_samples = int(
        math.ceil(max(content_dataset.n_samples, style_dataset.n_samples) / args.bs) * args.bs)
    content_dataset.n_samples = n_samples
    style_dataset.n_samples = n_samples
    content_loader = data.DataLoader(content_dataset, batch_size=args.bs, shuffle=True,
                                          num_workers=args.num_workers,
                                          pin_memory=args.pin_memory)
    print('content dataloader created.')
    style_loader = data.DataLoader(style_dataset, batch_size=args.bs, shuffle=True,
                                        num_workers=args.num_workers,
                                        pin_memory=args.pin_memory)
    print('style dataloader created.')
    return scratch, scratch_raw, content_loader, style_loader


if __name__ == "__main__":

    def getcolormap():
        from matplotlib.colors import ListedColormap
        colorlist = np.round(
            np.array([[0, 0, 0], [186, 137, 120], [124, 121, 174], [240, 216, 152], [148, 184, 216]]) / 256, decimals=2)
        mycolormap = ListedColormap(colors=colorlist, name='mycolor', N=5)
        return mycolormap


    import matplotlib.pyplot as plt
    bssfp_generator = DataGenerator(phase='train', modality='bssfp', crop_size=224, n_samples=1000, augmentation=True,
                                    data_dir='F:/data/mscmrseg/origin')
    for img, msk in bssfp_generator:
        print(img.shape, msk.shape)
        print(img.min(), img.max())
        print(np.argmax(msk, axis=-3).min(), np.argmax(msk, axis=-3).max())
        f, plots = plt.subplots(1, 2)
        plots[0].axis('off')
        plots[1].axis('off')
        plots[0].imshow(img[1], cmap='gray')
        plots[1].imshow(np.argmax(msk, axis=0), cmap=getcolormap(), vmin=0, vmax=3)
        plt.show()
        pass
