import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from pathlib import Path
from warnings import warn
import yaml
import torch

from data.data_loader import load_partition_data_coco
from utils.general import (
    labels_to_class_weights,
    increment_path,
    check_file,
    check_img_size,
)
from utils.general import intersect_dicts
from model.yolov5.models.yolo import Model as YOLOv5
from model.yolov6.yolov6.models.yolo import Model as YOLOv6
from model.yolov6.yolov6.utils.config import Config
from model.yolov6.yolov6.models.yolo import build_model as build_yolov6

from trainer.yolov5_trainer import YOLOv5Trainer
from trainer.yolov6_trainer import YOLOv6Trainer

try:
    import wandb
except ImportError:
    wandb = None
    logging.info(
        "Install Weights & Biases for experiment logging via 'pip install wandb' (recommended)"
    )


def init_yolo(args, device="cpu"):
    # init settings
    args.yolo_hyp = args.yolo_hyp or (
        "hyp.finetune.yaml" if args.weights else "hyp.scratch.yaml"
    )
    args.data_conf, args.yolo_cfg, args.yolo_hyp = (
        check_file(args.data_conf),
        check_file(args.yolo_cfg),
        check_file(args.yolo_hyp),
    )  # check files
    assert len(args.yolo_cfg) or len(
        args.weights
    ), "either yolo_cfg or weights must be specified"
    args.img_size.extend(
        [args.img_size[-1]] * (2 - len(args.img_size))
    )  # extend to 2 sizes (train, test)
    # args.name = "evolve" if args.evolve else args.name
    args.save_dir = increment_path(
        Path(args.project) / args.name, exist_ok=args.exist_ok
    )  # increment run

    # Hyperparameters
    with open(args.yolo_hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps
        if "box" not in hyp:
            warn(
                'Compatibility: %s missing "box" which was renamed from "giou" in %s'
                % (args.yolo_hyp, "https://github.com/ultralytics/yolov5/pull/1120")
            )
            hyp["box"] = hyp.pop("giou")

    args.total_batch_size = args.batch_size

    logging.info(f"Hyperparameters {hyp}")
    save_dir, epochs, batch_size, total_batch_size, weights = (
        Path(args.save_dir),
        args.epochs,
        args.batch_size,
        args.total_batch_size,
        args.weights,
    )

    # Directories
    wdir = save_dir / "weights"
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / "last.pt"
    best = wdir / "best.pt"
    results_file = save_dir / "results.txt"

    args.last, args.best, args.results_file = last, best, results_file

    # Configure
    with open(args.data_conf) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict
    train_path = data_dict["train"]
    test_path = data_dict["val"]
    nc, names = (
        (1, ["item"]) if args.single_cls else (int(data_dict["nc"]), data_dict["names"])
    )  # number classes, names
    assert len(names) == nc, "%g names found for nc=%g dataset in %s" % (
        len(names),
        nc,
        args.data,
    )  # check
    args.nc = nc  # change nc to actual number of classes

    # Model
    # print("weights:", weights)

    if args.model.lower() == "yolov5":
        pretrained = weights.endswith(".pt")
        if pretrained:
            ckpt = torch.load(weights, map_location=device)  # load checkpoint
            if hyp.get("anchors"):
                ckpt["model"].yaml["anchors"] = round(
                    hyp["anchors"]
                )  # force autoanchor
            model = YOLOv5(args.yolo_cfg or ckpt["model"].yaml, ch=3, nc=nc).to(
                device
            )  # create
            exclude = (
                ["anchor"] if args.yolo_cfg or hyp.get("anchors") else []
            )  # exclude keys
            state_dict = ckpt["model"].float().state_dict()  # to FP32
            state_dict = intersect_dicts(
                state_dict, model.state_dict(), exclude=exclude
            )  # intersect
            model.load_state_dict(state_dict, strict=False)  # load
            logging.info(
                "Transferred %g/%g items from %s"
                % (len(state_dict), len(model.state_dict()), weights)
            )  # report
        else:
            model = YOLOv5(args.yolo_cfg, ch=3, nc=nc).to(device)  # create
    elif args.model.lower() == "yolov6":
        args.yolov6_cfg = Config.fromfile(args.yolo_cfg)
        model = build_yolov6(args.yolov6_cfg, num_classes=nc, device=device)
        pretrained = weights.endswith(".pt")
        if pretrained:  # finetune if pretrained model is set
            """Load weights from checkpoint file, only assign weights those layers' name and shape are match."""
            ckpt = torch.load(weights, map_location=device)
            state_dict = ckpt["model"].float().state_dict()
            model_state_dict = model.state_dict()
            state_dict = {
                k: v
                for k, v in state_dict.items()
                if k in model_state_dict and v.shape == model_state_dict[k].shape
            }
            model.load_state_dict(state_dict, strict=False)
            del ckpt, state_dict, model_state_dict
    print(model)

    dataset = load_partition_data_coco(args, hyp, model)
    [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ] = dataset

    args.model_stride = model.stride
    gs = int(max(model.stride))  # grid size (max stride)
    imgsz, imgsz_test = [
        check_img_size(x, gs) for x in args.img_size
    ]  # verify imgsz are gs-multiples

    hyp["cls"] *= nc / 80.0
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    # model.class_weights = labels_to_class_weights(train_data_global.dataset.labels, nc).to(
    # device
    # )  # attach class weights
    model.names = names
    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(
        round(nbs / total_batch_size), 1
    )  # accumulate loss before optimizing
    hyp["weight_decay"] *= total_batch_size * accumulate / nbs  # scale weight_decay
    # logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    # Save run settings
    with open(save_dir / "hyp.yaml", "w") as f:
        yaml.dump(hyp, f, sort_keys=False)
    # with open(save_dir / "opt.yaml", "w") as f:
    #     # save args as yaml
    #     yaml.dump(args.__dict__, f, sort_keys=False)

    args.hyp = hyp  # add hyperparameters
    args.wandb = wandb

    # Trainer
    if args.model == "yolov5":
        trainer = YOLOv5Trainer(model=model, args=args)
    elif args.model == "yolov6":
        trainer = YOLOv6Trainer(model=model, args=args)

    return model, dataset, trainer, args
