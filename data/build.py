import logging
import numpy as np
import operator
import json
import os
import torch.utils.data
from detectron2.data import DatasetCatalog
from detectron2.utils.comm import get_world_size
from detectron2.data.common import (
    DatasetFromList,
    MapDataset,
)
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.samplers import (
    InferenceSampler,
    RepeatFactorTrainingSampler,
    TrainingSampler,
)
from detectron2.data.build import (
    trivial_batch_collator,
    worker_init_reset_seed,
    get_detection_dataset_dicts,
    build_batch_data_loader,
)

from arteryseg.data.datasets.arterydata import (
    load_artery_train
)


"""
This file contains the default logic to build a dataloader for training or testing.

load_label_func: json -> List[dataset dict], each is single image with gt
load_unlabel_func: img_name -> List[dataset dict], each is d slices images with or without gt

dataset_mapper: dataset dict -> input data

sampler
"""

# uesed by supervised-only baseline trainer
def build_detection_semisup_train_loader(cfg, mapper=None):

    dataset_dicts = get_detection_dataset_dicts(
        cfg.DATASETS.TRAIN,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )

    # Divide into labeled and unlabeled sets according to supervision percentage
    # 分成label与unlabel，一个参数是artery个数
    label_dicts, unlabel_dicts = divide_label_unlabel(
        dataset_dicts,
        cfg.DATALOADER.SUP_PERCENT,
        cfg.DATALOADER.RANDOM_DATA_SEED,
        cfg.DATALOADER.RANDOM_DATA_SEED_PATH,
    )

    dataset = DatasetFromList(label_dicts, copy=False)

    if mapper is None:
        mapper = DatasetMapper(cfg, True)
    dataset = MapDataset(dataset, mapper)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))

    if sampler_name == "TrainingSampler":
        sampler = TrainingSampler(len(dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        repeat_factors = (
            RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
                label_dicts, cfg.DATALOADER.REPEAT_THRESHOLD
            )
        )
        sampler = RepeatFactorTrainingSampler(repeat_factors)
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))

    # list num of labeled and unlabeled
    logger.info("Number of training samples " + str(len(dataset)))
    logger.info("Supervision percentage " + str(cfg.DATALOADER.SUP_PERCENT))

    return build_batch_data_loader(
        dataset,
        sampler,
        cfg.SOLVER.IMS_PER_BATCH,
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )


# uesed by evaluation
def build_detection_test_loader(cfg, dataset_name, mapper=None):
    label_dicts = DatasetCatalog.get(dataset_name)
    
    dataset = DatasetFromList(label_dicts)
    
    # no artery filter
    if mapper is None:
        mapper = DatasetMapper(cfg, False)
    dataset = MapDataset(dataset, mapper)

    sampler = InferenceSampler(len(dataset))
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader


# uesed by unbiased teacher trainer
def build_detection_semisup_train_loader_two_crops(cfg, mapper_s, mapper_u):

        
    label_dicts, unlabel_dicts = DatasetCatalog.get(cfg.DATASETS.TRAIN[0])
    
    # num of sup artery, filter
    all_arterys = np.unique([os.path.basename(d['file_name'])[:3] for d in label_dicts])
    selected = np.random.choice(all_arterys, int(cfg.DATASETS.NUMSUPARTERY))
    
    label_dicts = [d for d in label_dicts if os.path.basename(d['file_name'])[:3] in selected]

    label_dataset = DatasetFromList(label_dicts, copy=False)
    # exclude the labeled set from unlabeled dataset
    unlabel_dataset = DatasetFromList(unlabel_dicts, copy=False)
    # include the labeled set in unlabel dataset
    # unlabel_dataset = DatasetFromList(dataset_dicts, copy=False)

    label_dataset = MapDataset(label_dataset, mapper_s)
    unlabel_dataset = MapDataset(unlabel_dataset, mapper_u)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))
    if sampler_name == "TrainingSampler":
        label_sampler = TrainingSampler(len(label_dataset))
        unlabel_sampler = TrainingSampler(len(unlabel_dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        raise NotImplementedError("{} not yet supported.".format(sampler_name))
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))
        
    return build_semisup_batch_data_loader_two_crop(
        (label_dataset, unlabel_dataset),
        (label_sampler, unlabel_sampler),
        cfg.SOLVER.IMG_PER_BATCH_LABEL,
        cfg.SOLVER.IMG_PER_BATCH_UNLABEL,
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )


# batch data loader
def build_semisup_batch_data_loader_two_crop(
    dataset,
    sampler,
    total_batch_size_label,
    total_batch_size_unlabel,
    *,
    aspect_ratio_grouping=False,
    num_workers=0
):
    world_size = get_world_size()
    assert (
        total_batch_size_label > 0 and total_batch_size_label % world_size == 0
    ), "Total label batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size_label, world_size
    )

    assert (
        total_batch_size_unlabel > 0 and total_batch_size_unlabel % world_size == 0
    ), "Total unlabel batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size_label, world_size
    )

    batch_size_label = total_batch_size_label // world_size
    batch_size_unlabel = total_batch_size_unlabel // world_size

    label_dataset, unlabel_dataset = dataset
    label_sampler, unlabel_sampler = sampler


    label_data_loader = torch.utils.data.DataLoader(
        label_dataset,
        sampler=label_sampler,
        num_workers=num_workers,
        batch_sampler=None,
        collate_fn=operator.itemgetter(
            0
        ),  # don't batch, but yield individual elements
        worker_init_fn=worker_init_reset_seed,
    )  # yield individual mapped dict
    unlabel_data_loader = torch.utils.data.DataLoader(
        unlabel_dataset,
        sampler=unlabel_sampler,
        num_workers=num_workers,
        batch_sampler=None,
        collate_fn=operator.itemgetter(
            0
        ),  # don't batch, but yield individual elements
        worker_init_fn=worker_init_reset_seed,
    )  # yield individual mapped dict
    
    return GroupedSemiSupDatasetTwoCrop(
        (label_data_loader, unlabel_data_loader),
        (batch_size_label, batch_size_unlabel),
    )


class GroupedSemiSupDatasetTwoCrop(torch.utils.data.IterableDataset):
    def __init__(self, dataset, batch_size):
        """
        Args:
            dataset: a tuple containing two iterable generators. （labeled and unlabeled data)
               Each element must be a dict with keys "width" and "height", which will be used
               to batch data.
            batch_size (int):
        """

        self.label_dataset, self.unlabel_dataset = dataset
        self.batch_size_label = batch_size[0]
        self.batch_size_unlabel = batch_size[1]

        self._label_buckets = []
        self._label_buckets_key = []
        self._unlabel_buckets = []
        self._unlabel_buckets_key = []

    def __iter__(self):
        label_bucket, unlabel_bucket = [], []
        for d_label, d_unlabel in zip(self.label_dataset, self.unlabel_dataset):
            # d is a tuple with len = 2
            # It's two images (same size) from the same image instance
            # d[0] is with strong augmentation, d[1] is with weak augmentation

            if len(label_bucket) != self.batch_size_label:
                label_bucket = self._label_buckets
                label_bucket.append(d_label[0])
                label_buckets_key = self._label_buckets_key
                label_buckets_key.append(d_label[1])

            if len(unlabel_bucket) != self.batch_size_unlabel:
                unlabel_bucket = self._unlabel_buckets
                unlabel_bucket.append(d_unlabel[0])
                unlabel_buckets_key = self._unlabel_buckets_key
                unlabel_buckets_key.append(d_unlabel[1])

            # yield the batch of data until all buckets are full
            if (
                len(label_bucket) == self.batch_size_label
                and len(unlabel_bucket) == self.batch_size_unlabel
            ):
                # label_strong, label_weak, unlabed_strong, unlabled_weak
                ls, lw, us, uw = label_bucket[:], label_buckets_key[:], unlabel_bucket[:], unlabel_buckets_key[:]
                
                del label_bucket[:]
                del label_buckets_key[:]
                del unlabel_bucket[:]
                del unlabel_buckets_key[:]
                
                yield (ls, lw, us, uw)
        
