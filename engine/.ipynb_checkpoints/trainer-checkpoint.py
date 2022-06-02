# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import time
import logging
import torch
import torch.nn.functional as F
import copy
from torch.nn.parallel import DistributedDataParallel
from fvcore.nn.precise_bn import get_bn_modules
import numpy as np

from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer, SimpleTrainer, TrainerBase
from detectron2.engine.train_loop import AMPTrainer
from detectron2.utils.events import EventStorage
from detectron2.evaluation import COCOEvaluator, verify_results, PascalVOCDetectionEvaluator, DatasetEvaluators
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.engine import hooks
from detectron2.structures import Boxes, BitMasks
from detectron2.structures.instances import Instances
from detectron2.utils.env import TORCH_VERSION
from detectron2.data import MetadataCatalog

from arteryseg.data.build import (
    build_detection_semisup_train_loader,
    build_detection_test_loader,
    build_detection_semisup_train_loader_two_crops,
)
from arteryseg.data.dataset_mapper import (
    DatasetMapperTwoCropLabel,
    DatasetMapperTwoCropUnlabel
)
from arteryseg.engine.hooks import LossEvalHook
from arteryseg.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from arteryseg.checkpoint.detection_checkpoint import DetectionTSCheckpointer
from arteryseg.solver.build import build_lr_scheduler

from arteryseg.modeling.utils import IoU, vis_box

# Supervised-only Trainer
class BaselineTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
        if isinstance(self.model, DistributedDataParallel):
            # broadcast loaded data/model from the first rank, because other
            # machines may not have access to the checkpoint file
            if TORCH_VERSION >= (1, 7):
                self.model._sync_params_and_buffers()
            self.start_iter = comm.all_gather(self.start_iter)[0]

    def train_loop(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    def run_step(self):
        self._trainer.iter = self.iter

        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        # data:batch_sizeximg -> (small)batch_size x l imgs
        data = next(self._trainer._data_loader_iter) # 调取数据，batch
        data_time = time.perf_counter() - start

        record_dict, _, _, _ = self.model(data, branch="supervised") # supervised branch

        num_gt_bbox = 0.0
        for element in data:
            num_gt_bbox += len(element["instances"])
        num_gt_bbox = num_gt_bbox / len(data)
        record_dict["bbox_num/gt_bboxes"] = num_gt_bbox

        loss_dict = {}
        for key in record_dict.keys():
            if key[:4] == "loss" and key[-3:] != "val":
                loss_dict[key] = record_dict[key]

        losses = sum(loss_dict.values())

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(
                dataset_name, output_dir=output_folder))
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_semisup_train_loader(cfg, mapper=None)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable
        """
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                cfg.TEST.EVAL_PERIOD,
                self.model,
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    def _write_metrics(self, metrics_dict: dict):
        """
        Args:
            metrics_dict (dict): dict of scalar metrics
        """
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                data_time = np.max([x.pop("data_time")
                                   for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)


# Unbiased Teacher Trainer
class ArteryTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        data_loader = self.build_train_loader(cfg)

        # create an student model
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)

        # create an teacher model
        model_teacher = self.build_model(cfg)
        self.model_teacher = model_teacher

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        # Ensemble teacher and student model is for model saving and loading
        ensem_ts_model = EnsembleTSModel(model_teacher, model)

        self.checkpointer = DetectionTSCheckpointer(
            ensem_ts_model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
        if isinstance(self.model, DistributedDataParallel):
            # broadcast loaded data/model from the first rank, because other
            # machines may not have access to the checkpoint file
            if TORCH_VERSION >= (1, 7):
                self.model._sync_params_and_buffers()
            self.start_iter = comm.all_gather(self.start_iter)[0]

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = 'coco'
        #MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(
                dataset_name, output_dir=output_folder))
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper_s = DatasetMapperTwoCropLabel(cfg, True)
        mapper_u = DatasetMapperTwoCropUnlabel(cfg, True)
        return build_detection_semisup_train_loader_two_crops(cfg, mapper_s, mapper_u)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    def train(self):
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def train_loop(self, start_iter: int, max_iter: int):
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()

                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step_full_semisup() # training 核心函数
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    # =====================================================
    # ================== Pseduo-labeling ==================
    # =====================================================
    def threshold_bbox(self, proposal_bbox_inst, thres=0.7, proposal_type="roih"):
        if proposal_type == "rpn":
            valid_map = proposal_bbox_inst.objectness_logits > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
                valid_map
            ]
        elif proposal_type == "roih":
            valid_map = proposal_bbox_inst.scores > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
            new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]

        return new_proposal_inst

    def iou_pseudo_bbox(
        self, proposals_roi, prob_threshold, iou_threshold
    ):
        """
        在NMS之后
        proposals_roi: List[Instance], len=batch x depth
        return:
        new_proposals: List[Instance], len=n
        idx
        """
        k = self.cfg.SEMISUPNET.TOPK
        d = 2*self.cfg.DATASETS.DEPTH+1
        probs = [p.scores[:k] for p in proposals_roi]
        probs = [F.pad(p, (0, k-len(p))) for p in probs]
        probs = torch.stack(probs).view(-1, d, k) # b x d x topk
        bboxes = [p.pred_boxes.tensor[:k,:] for p in proposals_roi]
        bboxes = [F.pad(b, (0,0,0,k-len(b))) for b in bboxes]
        bboxes = torch.stack(bboxes).view(-1, d, k, 4) # b x d x topk x 4
        # class probability threshold
        if not (probs > prob_threshold).any():
            return None
        else:
            # iou threshold
            # todo: iou大于阈值的box都应该选为pseudo
            i_bs, i_ds, i_tks = torch.where(probs>prob_threshold)
            box_over_thre = bboxes[torch.where(probs>prob_threshold)].unsqueeze(1).unsqueeze(1) # n x 1 x 1 x 4
            bboxes_c = torch.stack([bboxes[i] for i in i_bs]) # n x d x topk x 4
            score = IoU(box_over_thre, bboxes_c) # reduce box, n x d x topk
            score, _ = score.max(axis=-1) # reduce topk dim, n x d
            score = (score.sum(axis=-1)-1) / (score.shape[-1]-1) # reduce d dim, n 
            # score: vector n, corresponding to box_over_thre
            # score > iou_threshold -> pseudo labels
            if not (score > iou_threshold).any():
                return None
            else:
                new_proposals = []
                ids = []
                first = True
                for i in range(len(score)):
                    if score[i] > iou_threshold:
                        # 会出现一张image,多个box吗
                        i_b, i_d, i_tk = i_bs[i], i_ds[i], i_tks[i]
                        inst = proposals_roi[i_b*d+i_d]
                        
                        new_box = bboxes[i_b, i_d, i_tk].unsqueeze(0)
                        new_c = inst.pred_classes[i_tk].unsqueeze(0)
                        new_inst = Instances(inst.image_size)
                        mask = torch.zeros(inst.image_size).unsqueeze(0)
                        if (i>0) and (not first) and (i_b*d+i_d == ids[-1]):
                            new_box = torch.cat([new_proposals[-1].gt_boxes.tensor, new_box])
                            mask = torch.cat([new_proposals[-1].gt_masks.tensor, mask])
                            
                            new_box = Boxes(new_box)
                            new_c = torch.cat([new_proposals[-1].gt_classes, new_c])
                            new_inst.gt_boxes = new_box
                            new_inst.gt_classes = new_c
                            new_inst.gt_masks = BitMasks(mask)
                            new_proposals[-1] = new_inst
                        else:
                            new_box = Boxes(new_box)
                            new_inst.gt_boxes = new_box
                            new_inst.gt_classes = new_c
                            new_inst.gt_masks = BitMasks(mask)
                            new_proposals.append(new_inst)
                            first = False
                        ids.append(i_b*d+i_d)
                            
                # return idx for image selection
                return new_proposals, ids
        
    
    def process_pseudo_label(
        self, proposals_rpn_unsup_k, cur_threshold, proposal_type, psedo_label_method=""
    ):
        list_instances = []
        num_proposal_output = 0.0
        for proposal_bbox_inst in proposals_rpn_unsup_k:
            # thresholding
            if psedo_label_method == "thresholding":
                proposal_bbox_inst = self.threshold_bbox(
                    proposal_bbox_inst, thres=cur_threshold, proposal_type="rpn"
                )
            else:
                raise ValueError("Unkown pseudo label boxes methods")
            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)
        num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)
        return list_instances, num_proposal_output
    
    def generate_pseudo_label(
        self, proposals_rpn_unsup_k, prob_threshold, iou_threshold, proposal_type, pseudo_label_method=""
    ):
        """
        在rpn层采用概率阈值的做法生成pseudo,借机将vein纳入foreground中，提高recall;
        在roi处采用概率和iou双阈值生成pseudo,形成artery/vein细分类的效果，提高precision；
        proposals_rpn_unsup_k: List[Instances], A list of N instances, one for each image in the batch that stores the topk most confidence detections.
        """
        list_instances = []
        num_proposal_output = 0.0
        if pseudo_label_method == 'threshold':
            # silce-wise pseudo label
            # 暂未确定
            for proposal_bbox_inst in proposals_rpn_unsup_k:
                proposal_bbox_inst = self.threshold_bbox(
                    proposal_bbox_inst, thre=prob_threshold, proposal_type='rpn'
                )
                num_proposal_output += len(proposal_bbox_inst)
                list_instances.append(proposal_bbox_inst)
        if pseudo_label_method == 'iouscore':
            # in fact graph-based pseudo label
            # temp_batch_size = true_batch_size x depth
            pseudos_and_idx = self.iou_pseudo_bbox(
                    proposals_rpn_unsup_k, prob_threshold, iou_threshold
                )
            if pseudos_and_idx is None:
                return None
            else:
                list_instances, idx = pseudos_and_idx
                num_proposal_output = sum([len(p) for p in list_instances]) / len(proposals_rpn_unsup_k)
                return list_instances, num_proposal_output, idx
    

    def remove_label(self, label_data):
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                del label_datum["instances"]
        return label_data

    def add_label(self, unlabled_data, label):
        for unlabel_datum, lab_inst in zip(unlabled_data, label):
            unlabel_datum["instances"] = lab_inst
        return unlabled_data

    # =====================================================
    # =================== Training Flow ===================
    # =====================================================

    def flatten_unlabel(self, unlabel_data):
        d = 2*self.cfg.DATASETS.DEPTH+1
        # 2.5d x batch_size -> 2d x (depth x batch_size), padding depth
        unlabel_data_new = []
        for data in unlabel_data:
            for i in range(d):
                data_new = copy.deepcopy(data)
                try:
                    image = data['image'][i]
                except:
                    image = data['image'][-1]
                data_new['image'] = image
                unlabel_data_new.append(data_new)
        return unlabel_data_new
                
        
    def run_step_full_semisup(self):
        self._trainer.iter = self.iter
        assert self.model.training, "[ArteryTeacherTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter) # get data
        # data_q and data_k from different augmentations (q:strong, k:weak)
        # label_strong, label_weak, unlabed_strong, unlabled_weak
        label_data_q, label_data_k, unlabel_data_q, unlabel_data_k = data
        data_time = time.perf_counter() - start
        #print(label_data_q[0]['image'].shape)
        # debug模式，间隔一定轮数，存储数据
        if self.cfg.DEBUG.DEBUG_IMAGE and (self.iter+1) % self.cfg.DEBUG.LOG_PERIOD == 0:
            #events = get_event_storage()
            first_image = label_data_q[0]['image'] # channel first
            img = copy.deepcopy(first_image).permute([1,2,0]).numpy()
            box = label_data_q[0]['instances'].gt_boxes.tensor.numpy()
            viz_gt = vis_box(img, box, (255,255,0)) # 黄色
            viz_gt = viz_gt[np.newaxis,:,:].repeat(3,0)
            self.storage.put_image('org_image', first_image)
            self.storage.put_image('gt_image', viz_gt)
            _ = self.model(label_data_q, branch="debug")   

        # remove unlabeled data labels
        # reshape unlabel data
        unlabel_data_q = self.remove_label(unlabel_data_q)
        unlabel_data_k = self.remove_label(unlabel_data_k)
        unlabel_data_q = self.flatten_unlabel(unlabel_data_q)
        unlabel_data_k = self.flatten_unlabel(unlabel_data_k)

        # burn-in stage (supervised training with labeled data)
        # 在半监督前先进行监督数据的update
        if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP:

            # input both strong and weak supervised data into model
            label_data_q.extend(label_data_k)
            record_dict, _, _, _ = self.model(
                label_data_q, branch="supervised")

            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = record_dict[key] * 1
            losses = sum(loss_dict.values())

        else:
            if self.iter == self.cfg.SEMISUPNET.BURN_UP_STEP:
                # update copy the the whole model
                self._update_teacher_model(keep_rate=0.00) # 复制权重

            elif (
                self.iter - self.cfg.SEMISUPNET.BURN_UP_STEP
            ) % self.cfg.SEMISUPNET.TEACHER_UPDATE_ITER == 0:
                self._update_teacher_model(
                    keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE) # 一定间隔更新teacher model

            record_dict = {}
            #  generate the pseudo-label using teacher model
            # note that we do not convert to eval mode, as 1) there is no gradient computed in
            # teacher model and 2) batch norm layers are not updated as well
            with torch.no_grad():
                (
                    _,
                    proposals_rpn_unsup_k,
                    proposals_roih_unsup_k,
                    _,
                ) = self.model_teacher(unlabel_data_k, branch="unsup_data_weak")

            #  Pseudo-labeling
            # 同时为一个段添加pseudo labels
            cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD
            iou_threshold = self.cfg.SEMISUPNET.IOU_THRESHOLD
            
            #joint_proposal_dict = {}
            #joint_proposal_dict["proposals_rpn"] = proposals_rpn_unsup_k
            #(
            #    pesudo_proposals_rpn_unsup_k,
            #    nun_pseudo_bbox_rpn,
            #) = self.generate_pseudo_label(
            #    proposals_rpn_unsup_k, cur_threshold, iou_threshold, "rpn", "threshold"
            #)
            #joint_proposal_dict["proposals_pseudo_rpn"] = pesudo_proposals_rpn_unsup_k # rpn pseudo labels
            
            # Pseudo_labeling for ROI head (bbox location/objectness)
            pesudo_proposals_roih_unsup_k_and_idx = self.generate_pseudo_label(
                proposals_roih_unsup_k, cur_threshold, iou_threshold, "roih", "iouscore"
            ) # 这里的roih参数是无用的
            
            
            all_label_data = label_data_q + label_data_k

            record_all_label_data, _, _, _ = self.model(
                all_label_data, branch="supervised"
            )
            record_dict.update(record_all_label_data)
            
            if pesudo_proposals_roih_unsup_k_and_idx is not None:
                pesudo_proposals_roih_unsup_k, _, idx = pesudo_proposals_roih_unsup_k_and_idx
                #joint_proposal_dict["proposals_pseudo_roih"] = pesudo_proposals_roih_unsup_k

                #  add pseudo-label to unlabeled data
                # unlabel_data_q is a list of Dict
                idx = torch.unique(torch.stack(idx))
                unlabel_data_q = [unlabel_data_q[i] for i in idx]
                if not len(unlabel_data_q) == len(pesudo_proposals_roih_unsup_k):
                    print(len(unlabel_data_q), len(pesudo_proposals_roih_unsup_k), idx)
                unlabel_data_q = self.add_label(
                    unlabel_data_q, pesudo_proposals_roih_unsup_k
                )
                
                # visualization
                if len(self.storage._vis_data):
                    img = unlabel_data_q[0]['image'].permute([1,2,0]).cpu().numpy()
                    box = unlabel_data_q[0]['instances'].gt_boxes.tensor.cpu().numpy()
                    img = vis_box(img, box, (255,255,0))
                    img = img[np.newaxis,:,:].repeat(3,0)
                    self.storage.put_image('pseudo_box', img)
                    
                
                #unlabel_data_k = self.add_label(
                #    unlabel_data_k, joint_proposal_dict["proposals_pseudo_roih"]
                #)

                #all_unlabel_data = unlabel_data_q
                record_all_unlabel_data, _, _, _ = self.model(
                    unlabel_data_q, branch="supervised"
                )
                new_record_all_unlabel_data = {}
                for key in record_all_unlabel_data.keys():
                    new_record_all_unlabel_data[key + "_pseudo"] = record_all_unlabel_data[
                        key
                    ]
                record_dict.update(new_record_all_unlabel_data)

            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key[:4] == "loss":
                    if key == "loss_mask_pseudo":
                        # pseudo bbox regression <- 0
                        loss_dict[key] = record_dict[key] * 0
                    # loss不包含box regression这一项，但我们是包含的
                    # 基于seg微调的box pseudo, 仅基于cls prob或iou score会引入bias?
                    # 仅box会变成pseudo, mask不做pseudo

                    if key[-6:] == "pseudo":  # unsupervised loss
                        loss_dict[key] = (
                            record_dict[key] *
                            self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
                        )
                    else:  # supervised loss
                        loss_dict[key] = record_dict[key] * 1

            losses = sum(loss_dict.values())

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

    def _write_metrics(self, metrics_dict: dict):
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }

        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)
        # all_hg_dict = comm.gather(hg_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time")
                                   for x in all_metrics_dict])
                #with EventStorage(self.iter) as self.storage:
                self.storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            # append the list
            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)

    @torch.no_grad()
    def _update_teacher_model(self, keep_rate=0.996):
        if comm.get_world_size() > 1:
            student_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
        else:
            student_model_dict = self.model.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in self.model_teacher.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                    student_model_dict[key] *
                    (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        self.model_teacher.load_state_dict(new_teacher_dict)

    @torch.no_grad()
    def _copy_main_model(self):
        # initialize all parameters
        if comm.get_world_size() > 1:
            rename_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
            self.model_teacher.load_state_dict(rename_model_dict)
        else:
            self.model_teacher.load_state_dict(self.model.state_dict())

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_test_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results_student():
            self._last_eval_results_student = self.test(self.cfg, self.model)
            _last_eval_results_student = {
                k + "_student": self._last_eval_results_student[k]
                for k in self._last_eval_results_student.keys()
            }
            return _last_eval_results_student

        def test_and_save_results_teacher():
            self._last_eval_results_teacher = self.test(
                self.cfg, self.model_teacher)
            return self._last_eval_results_teacher

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                   test_and_save_results_teacher))
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                   test_and_save_results_student))
        

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=self.cfg.DEBUG.LOG_PERIOD))
        return ret