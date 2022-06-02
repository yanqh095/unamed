import logging
import torchvision.transforms as transforms
from arteryseg.data.transforms.augmentation_impl import (
    GaussianBlur,
)


def build_strong_augmentation(cfg, is_train):
    """
    Create a list of :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """

    logger = logging.getLogger(__name__)
    augmentation = []
    if is_train:
        # This is simialr to SimCLR https://arxiv.org/abs/2002.05709
        augmentation.append(
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        )
        augmentation.append(transforms.RandomGrayscale(p=0.2))
        augmentation.append(transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5))

        randcrop_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomErasing(
                    p=0.3, scale=(0.005, 0.02), ratio=(0.3, 3.3), value=0
                ),
                transforms.RandomErasing(
                    p=0.5, scale=(0.005, 0.05), ratio=(0.1, 6), value=0
                ),
                transforms.RandomErasing(
                    p=0.7, scale=(0.005, 0.08), ratio=(0.05, 8), value=0
                ),
                transforms.ToPILImage(),
            ]
        )
        #augmentation.append(randcrop_transform)

        logger.info("Augmentations used in training: " + str(augmentation))
    return transforms.Compose(augmentation)