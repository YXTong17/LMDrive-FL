#!/usr/bin/env python3
import argparse
import json
import logging
import os
import pickle
import socket
import time
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime

import lavis.tasks as tasks
import torch
import torch.nn as nn
import torchvision.utils
import yaml
from config import parse_arguments
from lavis.datasets.builders.carla_dataset_builder import CarlaDatasetBuilder
from lavis.datasets.data_utils import prepare_sample
from lavis.datasets.datasets.dataloader_utils import IterLoader
from lavis.tasks import *
from render import render, render_waypoints
from tensorboardX import SummaryWriter

# from timm.data import create_carla_dataset, create_carla_loader
from timm.data import AugMixDataset  # create_dataset,; create_loader,
from timm.data import FastCollateMixup, Mixup, create_carla_loader, resolve_data_config
from timm.loss import (
    JsdCrossEntropy,
    LabelSmoothingCrossEntropy,
    SoftTargetCrossEntropy,
)
from timm.models import (
    convert_splitbn_model,
    create_model,
    load_checkpoint,
    model_parameters,
    resume_checkpoint,
    safe_model_name,
)
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import *
from timm.utils import ApexScaler, NativeScaler
from torch.nn.parallel import DistributedDataParallel as NativeDDP

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, "autocast") is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb

    has_wandb = True
except ImportError:
    has_wandb = False

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger("train")


class WaypointL1Loss:
    def __init__(self, l1_loss=torch.nn.L1Loss):
        self.loss = l1_loss(reduction="none")
        self.weights = [
            0.1407441030399059,
            0.13352157985305926,
            0.12588535273178575,
            0.11775496498388233,
            0.10901991343009122,
            0.09952110967153563,
            0.08901438656870617,
            0.07708872007078788,
            0.06294267636589287,
            0.04450719328435308,
        ]

    def __call__(self, output, target):
        invaild_mask = target.ge(1000)
        output[invaild_mask] = 0
        target[invaild_mask] = 0
        loss = self.loss(output, target)  # shape: n, 12, 2
        loss = torch.mean(loss, (0, 2))  # shape: 12
        loss = loss * torch.tensor(self.weights, device=output.device)
        return torch.mean(loss)


class LAVLoss:
    def __init__(self):
        self.prob_criterion = nn.BCEWithLogitsLoss(reduction="none")
        self.loc_criterion = nn.L1Loss(reduction="none")
        self.ori_criterion = nn.L1Loss(reduction="none")
        self.box_criterion = nn.L1Loss(reduction="none")
        self.spd_criterion = nn.L1Loss(reduction="none")
        # self.loc_criterion = nn.SmoothL1Loss(reduction='none')
        # self.ori_criterion = nn.SmoothL1Loss(reduction='none')
        # self.box_criterion = nn.SmoothL1Loss(reduction='none')
        # self.spd_criterion = nn.SmoothL1Loss(reduction='none')

    def __call__(self, output, target):
        prob = target[:, :, 0:1]
        prob_mean = prob.mean()
        prob_mean = torch.maximum(prob_mean, torch.ones_like(prob_mean) * 1e-7)
        prob_det = torch.sigmoid(output[:, :, 0] * (1 - 2 * target[:, :, 0]))

        det_loss = (
            prob_det * self.prob_criterion(output[:, :, 0], target[:, :, 0])
        ).mean() / prob_det.mean()
        loc_loss = (
            prob * self.loc_criterion(output[:, :, 1:3], target[:, :, 1:3])
        ).mean() / prob_mean
        box_loss = (
            prob * self.box_criterion(output[:, :, 3:5], target[:, :, 3:5])
        ).mean() / prob_mean
        ori_loss = (
            prob * self.ori_criterion(output[:, :, 5:7], target[:, :, 5:7])
        ).mean() / prob_mean
        spd_loss = (
            prob * self.ori_criterion(output[:, :, 7:8], target[:, :, 7:8])
        ).mean() / prob_mean

        det_loss = 0.4 * det_loss + 0.2 * loc_loss + 0.2 * box_loss + 0.2 * ori_loss
        return det_loss, spd_loss


class MVTL1Loss:
    def __init__(self, weight=1, l1_loss=torch.nn.L1Loss):
        self.loss = l1_loss()
        self.weight = weight

    def __call__(self, output, target):
        target_1_mask = target[:, :, 0].ge(0.01)
        target_0_mask = target[:, :, 0].le(0.01)
        target_prob_1 = torch.masked_select(target[:, :, 0], target_1_mask)
        output_prob_1 = torch.masked_select(output[:, :, 0], target_1_mask)
        target_prob_0 = torch.masked_select(target[:, :, 0], target_0_mask)
        output_prob_0 = torch.masked_select(output[:, :, 0], target_0_mask)
        if target_prob_1.numel() == 0:
            loss_prob_1 = 0
        else:
            loss_prob_1 = self.loss(output_prob_1, target_prob_1)
        if target_prob_0.numel() == 0:
            loss_prob_0 = 0
        else:
            loss_prob_0 = self.loss(output_prob_0, target_prob_0)
        loss_1 = 0.5 * loss_prob_0 + 0.5 * loss_prob_1

        output_1 = output[target_1_mask][:][:, 1:7]
        target_1 = target[target_1_mask][:][:, 1:7]
        if target_1.numel() == 0:
            loss_2 = 0
        else:
            loss_2 = self.loss(target_1, output_1)

        # speed pred loss
        output_2 = output[target_1_mask][:][:, 7]
        target_2 = target[target_1_mask][:][:, 7]
        if target_2.numel() == 0:
            loss_3 = target_2.sum()  # torch.tensor([0.0]).cuda()
        else:
            loss_3 = self.loss(target_2, output_2)
        return 0.5 * loss_1 * self.weight + 0.5 * loss_2, loss_3


def main():
    setup_default_logging()
    args, args_text = parse_arguments()

    if args.log_wandb:
        if has_wandb:
            wandb.init(project=args.experiment, config=args)
        else:
            _logger.warning(
                "You've requested to log metrics to wandb but package not found. "
                "Metrics not being logged to wandb, try `pip install wandb`"
            )
    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1
    args.device = "cuda:0"
    args.world_size = 1
    args.rank = 0  # global rank
    _logger.info("Training with a single process on 1 GPUs.")
    assert args.rank >= 0

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    if args.amp:
        # `--amp` chooses native amp before apex (APEX ver not actively maintained)
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    if args.apex_amp and has_apex:
        use_amp = "apex"
    elif args.native_amp and has_native_amp:
        use_amp = "native"
    elif args.apex_amp or args.native_amp:
        _logger.warning(
            "Neither APEX or native Torch AMP is available, using float32. "
            "Install NVIDA apex or upgrade to PyTorch 1.6"
        )

    random_seed(args.seed, args.rank)

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_tf=args.bn_tf,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint,
        freeze_num=args.freeze_num,
    )

    data_config = resolve_data_config(
        vars(args), model=model, verbose=args.local_rank == 0
    )

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, "A split of 1 makes no sense"
        num_aug_splits = args.aug_splits

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    # move model to GPU, enable channels last layout if set
    model.cuda()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.torchscript:
        assert not use_amp == "apex", "Cannot use APEX AMP with torchscripted model"
        assert not args.sync_bn, "Cannot use SyncBatchNorm with torchscripted model"
        model = torch.jit.script(model)

    linear_scaled_lr = args.lr * args.batch_size * 1 / 512.0
    args.lr = linear_scaled_lr
    if args.with_backbone_lr:
        if args.local_rank == 0:
            _logger.info(
                "CNN backbone and transformer blocks using different learning rates!"
            )
        backbone_linear_scaled_lr = args.backbone_lr * args.batch_size * 1 / 512.0
        backbone_weights = []
        other_weights = []
        for name, weight in model.named_parameters():
            if "backbone" in name and "lidar" not in name:
                backbone_weights.append(weight)
            else:
                other_weights.append(weight)
        if args.local_rank == 0:
            _logger.info(
                "%d weights in the cnn backbone, %d weights in other modules"
                % (len(backbone_weights), len(other_weights))
            )
        optimizer = create_optimizer_v2(
            [
                {"params": other_weights},
                {"params": backbone_weights, "lr": backbone_linear_scaled_lr},
            ],
            **optimizer_kwargs(cfg=args),
        )
    else:
        optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == "apex":
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        loss_scaler = ApexScaler()
        if args.local_rank == 0:
            _logger.info("Using NVIDIA APEX AMP. Training in mixed precision.")
    elif use_amp == "native":
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            _logger.info("Using native Torch AMP. Training in mixed precision.")
    else:
        if args.local_rank == 0:
            _logger.info("AMP not enabled. Training in float32.")

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model,
            args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.local_rank == 0,
        )

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV2(
            model,
            decay=args.model_ema_decay,
            device="cpu" if args.model_ema_force_cpu else None,
        )
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)

    # setup learning rate schedule and starting epoch
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        _logger.info("Scheduled epochs: {}".format(num_epochs))

    # create the train and eval datasets
    if "carla" in args.dataset:

        collate_fn = None
        mixup_fn = None
        from argparse import Namespace

        from lavis.common.config import Config
        from lavis.common.registry import registry
        from lavis.runners.runner_base import RunnerBase

        # 创建Namespace 对象，并设置一些属性
        args_data = Namespace(
            cfg_path="/home/tyx/yjl/LMDrive-FL/LAVIS/lavis/projects/lmdrive/notice_llava15_visual_encoder_r50_seq40.yaml",
            options=None,
        )

        cfg = Config(args_data)

    from lavis.common.utils import now

    job_id = now()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)

    def get_runner_class(cfg):
        runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))
        return runner_cls

    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )

    loader_train = runner.dataloaders["train"]
    print(type(loader_train))
    loader_eval = runner.dataloaders["val"]
    print(type(loader_eval))

    # setup loss function
    if args.smoothing > 0:
        cls_loss = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        cls_loss = nn.CrossEntropyLoss()

    if args.smoothed_l1:
        l1_loss = torch.nn.SmoothL1Loss
    else:
        l1_loss = torch.nn.L1Loss

    train_loss_fns = {
        # "traffic": MVTL1Loss(1.0, l1_loss=l1_loss),
        "traffic": LAVLoss(),
        "waypoints": torch.nn.L1Loss(),
        "cls": cls_loss,
        "stop_cls": cls_loss,
    }
    validate_loss_fns = {
        # "traffic": MVTL1Loss(1.0, l1_loss=l1_loss),
        "traffic": LAVLoss(),
        "waypoints": torch.nn.L1Loss(),
        "cls": cls_loss,
        "stop_cls": cls_loss,
    }

    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    writer = None
    if args.rank == 0:
        if args.experiment:
            exp_name = "-".join(
                [
                    datetime.now().strftime("%Y%m%d-%H%M%S"),
                    safe_model_name(args.model),
                    str(data_config["input_size"][-1]),
                    args.experiment,
                ]
            )
        else:
            exp_name = "-".join(
                [
                    datetime.now().strftime("%Y%m%d-%H%M%S"),
                    safe_model_name(args.model),
                    str(data_config["input_size"][-1]),
                ]
            )
        output_dir = get_outdir(args.output if args.output else "./output", exp_name)
        writer = SummaryWriter(logdir=output_dir)
        decreasing = args.saver_decreasing
        saver = CheckpointSaver(
            model=model,
            optimizer=optimizer,
            args=args,
            model_ema=model_ema,
            amp_scaler=loss_scaler,
            checkpoint_dir=output_dir,
            recovery_dir=output_dir,
            decreasing=decreasing,
            max_history=args.checkpoint_hist,
        )
        with open(os.path.join(output_dir, "args.yaml"), "w") as f:
            f.write(args_text)

    try:
        for epoch in range(start_epoch, num_epochs):
            if args.distributed and hasattr(loader_train.sampler, "set_epoch"):
                loader_train.sampler.set_epoch(epoch)

            train_metrics = train_one_epoch(
                epoch,
                model,
                loader_train,
                optimizer,
                train_loss_fns,
                args,
                writer,
                lr_scheduler=lr_scheduler,
                saver=saver,
                output_dir=output_dir,
                amp_autocast=amp_autocast,
                loss_scaler=loss_scaler,
                model_ema=model_ema,
                mixup_fn=mixup_fn,
            )

            eval_metrics = validate(
                epoch,
                model,
                loader_eval,
                validate_loss_fns,
                args,
                writer,
                amp_autocast=amp_autocast,
            )

            if model_ema is not None and not args.model_ema_force_cpu:
                if args.distributed and args.dist_bn in ("broadcast", "reduce"):
                    distribute_bn(model_ema, args.world_size, args.dist_bn == "reduce")
                ema_eval_metrics = validate(
                    model_ema.module,
                    loader_eval,
                    validate_loss_fns,
                    args,
                    amp_autocast=amp_autocast,
                    log_suffix=" (EMA)",
                )
                eval_metrics = ema_eval_metrics

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            if output_dir is not None:
                update_summary(
                    epoch,
                    train_metrics,
                    eval_metrics,
                    os.path.join(output_dir, "summary.csv"),
                    write_header=best_metric is None,
                    log_wandb=args.log_wandb and has_wandb,
                )

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(
                    epoch, metric=save_metric
                )

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        _logger.info("*** Best metric: {0} (epoch {1})".format(best_metric, best_epoch))


def retransform(data):
    std_tensor = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).cuda()
    mean_tensor = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).cuda()
    data = data * std_tensor + mean_tensor
    return data


def train_one_epoch(
    epoch,
    model,
    loader,
    optimizer,
    loss_fns,
    args,
    writer,
    lr_scheduler=None,
    saver=None,
    output_dir=None,
    amp_autocast=suppress,
    loss_scaler=None,
    model_ema=None,
    mixup_fn=None,
):

    second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    losses_waypoints = AverageMeter()
    losses_waypoints_llm = AverageMeter()  # 添加llm的waypoints losses记录
    losses_traffic = AverageMeter()
    losses_velocity = AverageMeter()
    losses_traffic_light_state = AverageMeter()
    losses_stop_sign = AverageMeter()

    model.train()
    end = time.time()

    #####

    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)

    for i in range(len(loader)):
        # if i % 100 == 0:
        #     print("当前batch index ：",i)
        last_batch = i == last_idx
        samples = next(loader)

        if samples == None:
            break
        samples = prepare_sample(samples, cuda_enabled=True)
        batch_size = samples["rgb_front"].size(0)
        # 将drive model里面的处理采样数据格式的代码转移进来

        target_visionencoder = samples["target"]
        # print(target_visionencoder)

        if True:
            with amp_autocast():
                input = samples
                output = model(input)
            target = target_visionencoder
            # print(target)
            # print(target[0].size())
            for t in range(len(target)):

                if t in [1, 4]:
                    target[t] = torch.reshape(
                        target[t],
                        (
                            target[t].size(1) * target[t].size(0),
                            target[t].size(2),
                            target[t].size(3),
                        ),
                    )
                elif t in [2]:
                    target[t] = torch.reshape(
                        target[t],
                        (target[t].size(1) * target[t].size(0), target[t].size(2)),
                    )
                elif t in [3, 6]:
                    target[t] = torch.reshape(
                        target[t], (target[t].size(1) * target[t].size(0),)
                    )

            # *
            """ 这里调用服务端的函数得到前向传播的来自大模型的waypoints_loss
            需要传入server model的数据如下:

            client model返回的参数变量output[6]: device bs_llm t num_features
            client model返回的output[5]: image_embeds
            server model需要的采样: x['text_input']    x['valid_frames'] x['local_future_waypoints']
                                   如果use_extra_prompt == True: x['text_before_img'] ['text_after_img']
                                   如果use_notice_prompt== True: x['notice_frame_id']
                                   如果has_gru_decoder  == True: x['target_point'] 
            
            
            """
            ServerModel_Input = {
                "text_input": input["text_input"],
                "valid_frames": input["valid_frames"],
                "local_future_waypoints": input["local_future_waypoints"],
                "text_before_img": input["text_before_img"],
                "text_after_img": input["text_after_img"],
                "notice_frame_id": input["notice_frame_id"],
                "target_point": input["target_point"],
                "image_embeds": output[5],
                "device": output[6][0],
                "bs_llm": output[6][1],
                "t": output[6][2],
                "num_features": model.num_features,
            }

            def Client_ContinueToForword_Server(ServerModel_Input):
                Client_to_Server(ServerModel_Input)
                return Server_to_Client()

            ServerOutput = Client_ContinueToForword_Server(
                ServerModel_Input
            )  # 客户端传递x给服务端，继续前向传播
            # 获得返回结果ServerOutput
            predicted_waypoints = ServerOutput
            gt_waypoints = build_gt_waypoints(
                input["local_future_waypoints"], input["valid_frames"]
            )
            waypoints_llm_loss = torch.nn.L1Loss()
            waypoints_loss_llm = waypoints_llm_loss(predicted_waypoints, gt_waypoints)

            # 梯度 = 计算梯度函数(predicted_waypoints , waypoints_loss_llm)
            # 梯度传输函数(梯度)
            # 梯度 = 得到服务端的梯度
            # 客户端反向传播()

            # *
            loss_traffic, loss_velocity = loss_fns["traffic"](output[0], target[4])

            loss_waypoints = loss_fns["waypoints"](output[1], target[1])
            on_road_mask = target[2] < 0.5

            loss_traffic_light_state = loss_fns["cls"](output[2], target[3])
            loss_stop_sign = loss_fns["stop_cls"](output[3], target[6])

            loss = (
                loss_traffic * 0.5
                + loss_waypoints * 0.5
                + loss_velocity * 0.05
                + loss_traffic_light_state * 0.1
                + loss_stop_sign * 0.01
                + waypoints_loss_llm
            )

        if not args.distributed:
            losses_traffic.update(loss_traffic.item(), batch_size)
            losses_waypoints.update(loss_waypoints.item(), batch_size)
            losses_waypoints_llm.update(waypoints_loss_llm, batch_size)

            losses_m.update(loss.item(), batch_size)

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss,
                optimizer,
                clip_grad=args.clip_grad,
                clip_mode=args.clip_mode,
                parameters=model_parameters(
                    model, exclude_head="agc" in args.clip_mode
                ),
                create_graph=second_order,
            )
        else:
            if i % 100 == 0 and i != 0:
                print("当前调用backward，记录反向传播时间——")
                back_before = time.time()

            loss.backward(create_graph=second_order)
            if i % 100 == 0 and i != 0:
                print("当前反向传播时间：", time.time() - back_before)

            if args.clip_grad is not None:
                dispatch_clip_grad(
                    model_parameters(model, exclude_head="agc" in args.clip_mode),
                    value=args.clip_grad,
                    mode=args.clip_mode,
                )
            optimizer.step()

        if model_ema is not None:
            model_ema.update(model)

        torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)
        if last_batch or i % args.log_interval == 0:
            lrl = [param_group["lr"] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), batch_size)
                reduced_loss_traffic = reduce_tensor(loss_traffic.data, args.world_size)
                losses_traffic.update(reduced_loss_traffic.item(), batch_size)
                reduced_loss_velocity = reduce_tensor(
                    loss_velocity.data, args.world_size
                )
                losses_velocity.update(reduced_loss_velocity.item(), batch_size)

                reduced_loss_waypoints = reduce_tensor(
                    loss_waypoints.data, args.world_size
                )
                losses_waypoints.update(reduced_loss_waypoints.item(), batch_size)
                reduced_loss_traffic_light_state = reduce_tensor(
                    loss_traffic_light_state.data, args.world_size
                )
                losses_traffic_light_state.update(
                    reduced_loss_traffic_light_state.item(), batch_size
                )
                reduced_loss_stop_sign = reduce_tensor(
                    loss_stop_sign.data, args.world_size
                )
                losses_stop_sign.update(reduced_loss_stop_sign.item(), batch_size)
                if writer and args.local_rank == 0:
                    writer.add_scalar("train/loss", reduced_loss.item(), num_updates)
                    writer.add_scalar(
                        "train/loss_traffic", reduced_loss_traffic.item(), num_updates
                    )
                    writer.add_scalar(
                        "train/loss_velocity", reduced_loss_velocity.item(), num_updates
                    )
                    writer.add_scalar(
                        "train/loss_waypoints",
                        reduced_loss_waypoints.item(),
                        num_updates,
                    )
                    writer.add_scalar(
                        "train/loss_traffic_light_state",
                        reduced_loss_traffic_light_state.item(),
                        num_updates,
                    )
                    writer.add_scalar(
                        "train/loss_stop_sign",
                        reduced_loss_stop_sign.item(),
                        num_updates,
                    )

                    # Add Image
                    writer.add_image(
                        "train/front_view",
                        retransform(input["rgb_front"][0]),
                        num_updates,
                    )
                    writer.add_image(
                        "train/left_view",
                        retransform(input["rgb_left"][0]),
                        num_updates,
                    )
                    writer.add_image(
                        "train/right_view",
                        retransform(input["rgb_right"][0]),
                        num_updates,
                    )
                    writer.add_image(
                        "train/rear_view",
                        retransform(input["rgb_rear"][0]),
                        num_updates,
                    )
                    writer.add_image(
                        "train/front_center_view",
                        retransform(input["rgb_center"][0]),
                        num_updates,
                    )
                    writer.add_image(
                        "train/pred_traffic",
                        torch.clip(output[0][0], 0, 1).view(1, 50, 50, 8)[:, :, :, 0],
                        num_updates,
                    )
                    writer.add_image(
                        "train/pred_traffic_render",
                        torch.clip(
                            torch.tensor(
                                render(
                                    output[0][0].view(50, 50, 8).detach().cpu().numpy()
                                )[:250, 25:275]
                            ),
                            0,
                            255,
                        ).view(1, 250, 250),
                        num_updates,
                    )
                    # input["lidar"][0] = input["lidar"][0] / torch.max(input["lidar"][0])
                    # writer.add_image(
                    #    "train/lidar", torch.clip(input["lidar"][0], 0, 1), num_updates
                    # )
                    writer.add_image(
                        "train/gt_traffic",
                        torch.clip(target[4][0], 0, 1).view(1, 50, 50, 8)[:, :, :, 0],
                        num_updates,
                    )
                    writer.add_image(
                        "train/gt_highres_traffic",
                        torch.clip(target[0][0], 0, 1),
                        num_updates,
                    )
                    writer.add_image(
                        "train/pred_waypoints",
                        torch.clip(
                            torch.tensor(
                                render_waypoints(output[1][0].detach().cpu().numpy())[
                                    :250, 25:275
                                ]
                            ),
                            0,
                            255,
                        ).view(1, 250, 250),
                        num_updates,
                    )
                    writer.add_image(
                        "train/gt_waypoints",
                        torch.clip(target[5][0], 0, 1),
                        num_updates,
                    )

            if args.local_rank == 0:
                _logger.info(
                    "Train: {} [{:>4d}/{} ({:>3.0f}%)]  "
                    "Loss(traffic): {loss_traffic.val:>9.6f} ({loss_traffic.avg:>6.4f})  "
                    "Loss(waypoints): {loss_waypoints.val:>9.6f} ({loss_waypoints.avg:>6.4f})  "
                    "Loss(waypoints_llm): {loss_waypoints_llm.val:>9.6f} ({loss_waypoints_llm.avg:>6.4f})  "
                    "Loss(light): {loss_traffic_light_state.val:>9.6f} ({loss_traffic_light_state.avg:>6.4f})  "
                    "Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  "
                    "Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  "
                    "({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  "
                    "LR: {lr:.3e}  "
                    "Data: {data_time.val:.3f} ({data_time.avg:.3f})".format(
                        epoch,
                        i,
                        len(loader),
                        100.0 * i / last_idx,
                        loss=losses_m,
                        loss_traffic=losses_traffic,
                        loss_waypoints=losses_waypoints,
                        loss_waypoints_llm=losses_waypoints_llm,
                        loss_traffic_light_state=losses_traffic_light_state,
                        batch_time=batch_time_m,
                        rate=batch_size * args.world_size / batch_time_m.val,
                        rate_avg=batch_size * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m,
                    )
                )

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, "train-batch-%d.jpg" % i),
                        padding=0,
                        normalize=True,
                    )

            if (
                saver is not None
                and args.recovery_interval
                and (last_batch or (i + 1) % args.recovery_interval == 0)
            ):
                saver.save_recovery(epoch, i=i)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, "sync_lookahead"):
        optimizer.sync_lookahead()

    return OrderedDict([("loss", losses_m.avg)])


def validate(
    epoch, model, loader, loss_fns, args, writer, amp_autocast=suppress, log_suffix=""
):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    losses_waypoints = AverageMeter()
    losses_traffic = AverageMeter()
    losses_velocity = AverageMeter()
    losses_traffic_light_state = AverageMeter()
    losses_stop_sign = AverageMeter()

    l1_errorm = AverageMeter()
    traffic_light_state_errorm = AverageMeter()
    stop_sign_errorm = AverageMeter()

    losses_waypoints_llm = AverageMeter()  # 添加llm的waypoints losses记录

    loader = IterLoader(loader, use_distributed=False)  # 将loader改成train的形式

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        # 为了和上面一致，我把所有的batch_idx都用的i
        for i in range(len(loader)):
            # if i % 100 == 0:
            #     print("当前batch index ：",i)
            last_batch = i == last_idx
            samples = next(loader)

            if samples == None:
                break
            samples = prepare_sample(samples, cuda_enabled=True)
            batch_size = samples["rgb_front"].size(0)
            target_visionencoder = samples["target"]

            with amp_autocast():
                input = samples
                output = model(input)
            target = target_visionencoder
            # print(target)

            for t in range(len(target)):
                if t in [1, 4]:
                    target[t] = torch.reshape(
                        target[t],
                        (
                            target[t].size(1) * target[t].size(0),
                            target[t].size(2),
                            target[t].size(3),
                        ),
                    )
                elif t in [2]:
                    target[t] = torch.reshape(
                        target[t],
                        (target[t].size(1) * target[t].size(0), target[t].size(2)),
                    )
                elif t in [3, 6]:
                    target[t] = torch.reshape(
                        target[t], (target[t].size(1) * target[t].size(0),)
                    )
                elif t in [0, 5]:
                    target[t] = torch.reshape(
                        target[t],
                        (
                            target[t].size(1),
                            target[t].size(0),
                            target[t].size(2),
                            target[t].size(3),
                        ),
                    )

                    # *
            """ 这里调用服务端的函数得到前向传播的来自大模型的waypoints_loss
            需要传入server model的数据如下:

            client model返回的参数变量output[6]: device bs_llm t num_features
            client model返回的output[5]: image_embeds
            server model需要的采样: x['text_input']    x['valid_frames'] x['local_future_waypoints']
                                   如果use_extra_prompt == True: x['text_before_img'] ['text_after_img']
                                   如果use_notice_prompt== True: x['notice_frame_id']
                                   如果has_gru_decoder  == True: x['target_point'] 
            
            
            """
            ServerModel_Input = {
                "text_input": input["text_input"],
                "valid_frames": input["valid_frames"],
                "local_future_waypoints": input["local_future_waypoints"],
                "text_before_img": input["text_before_img"],
                "text_after_img": input["text_after_img"],
                "notice_frame_id": input["notice_frame_id"],
                "target_point": input["target_point"],
                "image_embeds": output[5],
                "device": output[6][0],
                "bs_llm": output[6][1],
                "t": output[6][2],
                "num_features": model.num_features,
            }

            def Client_ContinueToForword_Server(ServerModel_Input):
                Client_to_Server(ServerModel_Input)
                return Server_to_Client()

            ServerOutput = Client_ContinueToForword_Server(
                ServerModel_Input
            )  # 客户端传递x给服务端，继续前向传播
            # 获得返回结果ServerOutput
            predicted_waypoints = ServerOutput
            gt_waypoints = build_gt_waypoints(
                input["local_future_waypoints"], input["valid_frames"]
            )
            waypoints_llm_loss = torch.nn.L1Loss()
            waypoints_loss_llm = waypoints_llm_loss(predicted_waypoints, gt_waypoints)

            # 梯度 = 计算梯度函数(predicted_waypoints , waypoints_loss_llm)
            # 梯度传输函数(梯度)
            # 梯度 = 得到服务端的梯度
            # 客户端反向传播()

            # *
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0 : target.size(0) : reduce_factor]

            loss_traffic, loss_velocity = loss_fns["traffic"](output[0], target[4])
            loss_waypoints = loss_fns["waypoints"](output[1], target[1])
            on_road_mask = target[2] < 0.5
            loss_traffic_light_state = loss_fns["cls"](output[2], target[3])
            loss_stop_sign = loss_fns["stop_cls"](output[3], target[6])
            loss = (
                loss_traffic * 0.5
                + loss_waypoints * 0.5
                + loss_velocity * 0.05
                + loss_traffic_light_state * 0.1
                + loss_stop_sign * 0.01
                + waypoints_loss_llm * 0.2
            )

            traffic_light_state_error = accuracy(output[2], target[3])[0]
            stop_sign_error = accuracy(output[3], target[6])[0]

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                reduced_loss_traffic = reduce_tensor(loss_traffic.data, args.world_size)
                reduced_loss_velocity = reduce_tensor(
                    loss_velocity.data, args.world_size
                )
                reduced_loss_waypoints = reduce_tensor(
                    loss_waypoints.data, args.world_size
                )
                reduced_loss_traffic_light_state = reduce_tensor(
                    loss_traffic_light_state.data, args.world_size
                )
                reduced_loss_stop_sign = reduce_tensor(
                    loss_stop_sign.data, args.world_size
                )
                reduced_traffic_light_state_error = reduce_tensor(
                    traffic_light_state_error, args.world_size
                )
                reduced_stop_sign_error = reduce_tensor(
                    stop_sign_error, args.world_size
                )
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()
            if not args.distributed:  # 补充
                losses_traffic.update(loss_traffic.item(), batch_size)
                losses_waypoints.update(loss_waypoints.item(), batch_size)
                losses_waypoints_llm.update(output[5], batch_size)
                losses_m.update(loss.item(), batch_size)

            if args.distributed:
                losses_m.update(reduced_loss.item(), batch_size)
                losses_traffic.update(reduced_loss_traffic.item(), batch_size)
                losses_velocity.update(reduced_loss_velocity.item(), batch_size)
                losses_waypoints.update(reduced_loss_waypoints.item(), batch_size)
                losses_traffic_light_state.update(
                    reduced_loss_traffic_light_state.item(), batch_size
                )
                losses_stop_sign.update(reduced_loss_stop_sign.item(), batch_size)

                l1_errorm.update(reduced_loss.item(), batch_size)
                traffic_light_state_errorm.update(
                    reduced_traffic_light_state_error.item(), batch_size
                )
                stop_sign_errorm.update(reduced_stop_sign_error.item(), batch_size)

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (last_batch or i % args.log_interval == 0):
                log_name = "Test" + log_suffix
                _logger.info(
                    "{0}: [{1:>4d}/{2}]  "
                    "Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  "
                    "Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  "
                    "Loss(traffic): {loss_traffic.val:>7.4f} ({loss_traffic.avg:>6.4f})  "
                    "Loss(waypoints): {loss_waypoints.val:>7.4f} ({loss_waypoints.avg:>6.4f})  "
                    "Loss(waypoints_llm): {loss_waypoints_llm.val:>9.6f} ({loss_waypoints_llm.avg:>6.4f})  "
                    "Loss(light): {loss_traffic_light_state.val:>9.6f} ({loss_traffic_light_state.avg:>6.4f})  "
                    "Acc(light): {traffic_light_state_errorm.val:>9.6f} ({traffic_light_state_errorm.avg:>6.4f})  ".format(
                        log_name,
                        i,
                        last_idx,
                        batch_time=batch_time_m,
                        loss_traffic_light_state=losses_traffic_light_state,
                        loss_waypoints_llm=losses_waypoints_llm,
                        traffic_light_state_errorm=traffic_light_state_errorm,
                        loss=losses_m,
                        loss_traffic=losses_traffic,
                        loss_waypoints=losses_waypoints,
                    )
                )
                if writer:
                    # Add Image
                    writer.add_image(
                        "val/%d_front_view" % i,
                        retransform(input["rgb_front"][0]),
                        epoch,
                    )
                    writer.add_image(
                        "val/%d_left_view" % i,
                        retransform(input["rgb_left"][0]),
                        epoch,
                    )
                    writer.add_image(
                        "val/%d_right_view" % i,
                        retransform(input["rgb_right"][0]),
                        epoch,
                    )
                    writer.add_image(
                        "val/%d_front_center_view" % i,
                        retransform(input["rgb_center"][0]),
                        epoch,
                    )
                    writer.add_image(
                        "val/%d_rear_view" % i,
                        retransform(input["rgb_rear"][0]),
                        epoch,
                    )
                    writer.add_image(
                        "val/%d_pred_traffic" % i,
                        torch.clip(output[0][0], 0, 1).view(1, 50, 50, 8)[:, :, :, 0],
                        epoch,
                    )
                    writer.add_image(
                        "val/%d_gt_traffic" % i,
                        torch.clip(target[4][0], 0, 1).view(1, 50, 50, 8)[:, :, :, 0],
                        epoch,
                    )
                    writer.add_image(
                        "val/%d_highres_gt_traffic" % i,
                        torch.clip(target[0][0], 0, 1),
                        epoch,
                    )

                    writer.add_image(
                        "val/%d_gt_waypoints" % i,
                        torch.clip(target[5][0], 0, 1),
                        epoch,
                    )
                    writer.add_image(
                        "val/%d_pred_traffic_render" % i,
                        torch.clip(
                            torch.tensor(
                                render(
                                    output[0][0].view(50, 50, 8).detach().cpu().numpy()
                                )[:250, 25:275]
                            ),
                            0,
                            255,
                        ).view(1, 250, 250),
                        epoch,
                    )
                    writer.add_image(
                        "val/%d_pred_waypoints" % i,
                        torch.clip(
                            torch.tensor(
                                render_waypoints(output[1][0].detach().cpu().numpy())[
                                    :250, 25:275
                                ]
                            ),
                            0,
                            255,
                        ).view(1, 250, 250),
                        epoch,
                    )

        if writer:
            writer.add_scalar("val/loss", losses_m.avg, epoch)
            writer.add_scalar("val/loss_traffic", losses_traffic.avg, epoch)
            writer.add_scalar("val/loss_velocity", losses_velocity.avg, epoch)
            writer.add_scalar("val/loss_waypoints", losses_waypoints.avg, epoch)
            writer.add_scalar(
                "val/loss_traffic_light_state", losses_traffic_light_state.avg, epoch
            )
            writer.add_scalar("val/loss_stop_sign", losses_stop_sign.avg, epoch)
            writer.add_scalar(
                "val/acc_traffic_light_state", traffic_light_state_errorm.avg, epoch
            )
            writer.add_scalar("val/acc_stop_sign", stop_sign_errorm.avg, epoch)

    metrics = OrderedDict([("loss", losses_m.avg), ("l1_error", l1_errorm.avg)])

    return metrics


def Server_to_Client():
    host, port = "0.0.0.0", 20841
    Receiver_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    Receiver_socket.bind((host, port))
    Receiver_socket.listen(1)
    print(f"Server listening on {host}:{port}")

    Sent_socket, addr = Receiver_socket.accept()
    print(f"Connection from {addr} has been established.")

    # 接收数据长度
    data_length = Sent_socket.recv(1024)
    data_length = int(data_length)

    # 初始化接收的数据缓冲区
    received_data = b""

    # 循环接收数据直到达到指定长度
    while len(received_data) < data_length:
        chunk = Sent_socket.recv(1024)
        if not chunk:
            break
        received_data += chunk

    # 使用pickle反序列化数据
    received_dict = pickle.loads(received_data)
    Sent_socket.close()
    Receiver_socket.close()
    return received_dict


def Client_to_Server(ServerOutput):
    Sent_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    time.sleep(10)
    Sent_socket.connect(("127.0.0.1", 20840))
    # 使用pickle序列化字典
    serialized_data = pickle.dumps(ServerOutput)

    # 发送数据长度
    Sent_socket.sendall(str(len(serialized_data)).encode("utf-8"))
    print("已经发送数据长度,", len(serialized_data))
    # 发送数据
    Sent_socket.sendall(serialized_data)

    # 接收服务端的响应
    response = Sent_socket.recv(1024)

    Sent_socket.close()


def build_gt_waypoints(waypoints, valid_frames):
    gt_waypoints = []
    for i in range(waypoints.size(0)):
        gt_waypoints.append(waypoints[i, : valid_frames[i]])
    gt_waypoints = torch.cat(gt_waypoints, dim=0)
    return gt_waypoints


if __name__ == "__main__":
    main()
