import argparse
import yaml

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below


def parse_arguments():
    config_parser = parser = argparse.ArgumentParser(
        description="Training Config", add_help=False
    )
    parser.add_argument(
        "-c",
        "--config",
        default="",
        type=str,
        metavar="FILE",
        help="YAML config file specifying default arguments",
    )

    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

    parser.add_argument(
        "--train-towns",
        type=int,
        nargs="+",
        default=[0],
        help="dataset train towns (default: [0])",
    )
    parser.add_argument(
        "--val-towns",
        type=int,
        nargs="+",
        default=[1],
        help="dataset validation towns (default: [1])",
    )
    parser.add_argument(
        "--train-weathers",
        type=int,
        nargs="+",
        default=[0],
        help="dataset train weathers (default: [0])",
    )
    parser.add_argument(
        "--val-weathers",
        type=int,
        nargs="+",
        default=[1],
        help="dataset validation weathers (default: [1])",
    )
    parser.add_argument(
        "--saver-decreasing",
        action="store_true",
        default=False,
        help="StarAt with pretrained version of specified network (if avail)",
    )
    parser.add_argument(
        "--with-lidar",
        action="store_true",
        default=False,
        help="load lidar data in the dataset",
    )
    parser.add_argument(
        "--with-seg",
        action="store_true",
        default=False,
        help="load segmentation data in the dataset",
    )
    parser.add_argument(
        "--with-depth",
        action="store_true",
        default=False,
        help="load depth data in the dataset",
    )
    parser.add_argument(
        "--multi-view",
        action="store_true",
        default=False,
        help="load multi-view data in the dataset",
    )
    parser.add_argument(
        "--multi-view-input-size",
        default=None,
        nargs=3,
        type=int,
        metavar="N N N",
        help="Input all image dimensions (d h w, e.g. --input-size 3 224 224) for left- right- or rear- view",
    )

    parser.add_argument(
        "--temporal-frames", type=int, default=1, help="Number of frames of the input"
    )
    parser.add_argument(
        "--freeze-num",
        type=int,
        default=-1,
        help="Number of freeze layers in the backbone",
    )
    parser.add_argument(
        "--backbone-lr", type=float, default=5e-4, help="The learning rate for backbone"
    )
    parser.add_argument(
        "--with-backbone-lr",
        action="store_true",
        default=False,
        help="The learning rate for backbone is set as backbone-lr",
    )

    # Dataset / Model parameters
    parser.add_argument("data_dir", metavar="DIR", help="path to dataset")
    parser.add_argument(
        "--dataset",
        "-d",
        metavar="NAME",
        default="newcarla",
        help="dataset type (default: ImageFolder/ImageTar if empty)",
    )
    parser.add_argument(
        "--train-split",
        metavar="NAME",
        default="train",
        help="dataset train split (default: train)",
    )
    parser.add_argument(
        "--val-split",
        metavar="NAME",
        default="validation",
        help="dataset validation split (default: validation)",
    )
    parser.add_argument(
        "--model",
        default="resnet101",
        type=str,
        metavar="MODEL",
        help='Name of model to train (default: "countception"',
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=False,
        help="Start with pretrained version of specified network (if avail)",
    )
    parser.add_argument(
        "--initial-checkpoint",
        default="",
        type=str,
        metavar="PATH",
        help="Initialize model from this checkpoint (default: none)",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="Resume full model and optimizer state from checkpoint (default: none)",
    )
    parser.add_argument(
        "--no-resume-opt",
        action="store_true",
        default=False,
        help="prevent resume of optimizer state when resuming model",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        metavar="N",
        help="number of label classes (Model default if None)",
    )
    parser.add_argument(
        "--gp",
        default=None,
        type=str,
        metavar="POOL",
        help="Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        metavar="N",
        help="Image patch size (default: None => model default)",
    )
    parser.add_argument(
        "--input-size",
        default=None,
        nargs=3,
        type=int,
        metavar="N N N",
        help="Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty",
    )
    parser.add_argument(
        "--crop-pct",
        default=None,
        type=float,
        metavar="N",
        help="Input image center crop percent (for validation only)",
    )
    parser.add_argument(
        "--mean",
        type=float,
        nargs="+",
        default=None,
        metavar="MEAN",
        help="Override mean pixel value of dataset",
    )
    parser.add_argument(
        "--std",
        type=float,
        nargs="+",
        default=None,
        metavar="STD",
        help="Override std deviation of of dataset",
    )
    parser.add_argument(
        "--interpolation",
        default="",
        type=str,
        metavar="NAME",
        help="Image resize interpolation type (overrides model)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for training (default: 32)",
    )
    parser.add_argument(
        "-vb",
        "--validation-batch-size-multiplier",
        type=int,
        default=1,
        metavar="N",
        help="ratio of validation batch size to training batch size (default: 1)",
    )
    parser.add_argument(
        "--augment-prob",
        type=float,
        default=0.0,
    )

    # Optimizer parameters
    parser.add_argument(
        "--opt",
        default="sgd",
        type=str,
        metavar="OPTIMIZER",
        help='Optimizer (default: "sgd"',
    )
    parser.add_argument(
        "--opt-eps",
        default=None,
        type=float,
        metavar="EPSILON",
        help="Optimizer Epsilon (default: None, use opt default)",
    )
    parser.add_argument(
        "--opt-betas",
        default=None,
        type=float,
        nargs="+",
        metavar="BETA",
        help="Optimizer Betas (default: None, use opt default)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="Optimizer momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0001,
        help="weight decay (default: 0.0001)",
    )
    parser.add_argument(
        "--clip-grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--clip-mode",
        type=str,
        default="norm",
        help='Gradient clipping mode. One of ("norm", "value", "agc")',
    )

    # Learning rate schedule parameters
    parser.add_argument(
        "--sched",
        default="cosine",
        type=str,
        metavar="SCHEDULER",
        help='LR scheduler (default: "step"',
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--lr-noise",
        type=float,
        nargs="+",
        default=None,
        metavar="pct, pct",
        help="learning rate noise on/off epoch percentages",
    )
    parser.add_argument(
        "--lr-noise-pct",
        type=float,
        default=0.67,
        metavar="PERCENT",
        help="learning rate noise limit percent (default: 0.67)",
    )
    parser.add_argument(
        "--lr-noise-std",
        type=float,
        default=1.0,
        metavar="STDDEV",
        help="learning rate noise std-dev (default: 1.0)",
    )
    parser.add_argument(
        "--lr-cycle-mul",
        type=float,
        default=1.0,
        metavar="MULT",
        help="learning rate cycle len multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--lr-cycle-limit",
        type=int,
        default=1,
        metavar="N",
        help="learning rate cycle limit",
    )
    parser.add_argument(
        "--warmup-lr",
        type=float,
        default=5e-6,
        metavar="LR",
        help="warmup learning rate (default: 0.0001)",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-5,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0 (1e-5)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        metavar="N",
        help="number of epochs to train (default: 2)",
    )
    parser.add_argument(
        "--epoch-repeats",
        type=float,
        default=0.0,
        metavar="N",
        help="epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).",
    )
    parser.add_argument(
        "--start-epoch",
        default=None,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "--decay-epochs",
        type=float,
        default=30,
        metavar="N",
        help="epoch interval to decay LR",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        metavar="N",
        help="epochs to warmup LR, if scheduler supports",
    )
    parser.add_argument(
        "--cooldown-epochs",
        type=int,
        default=10,
        metavar="N",
        help="epochs to cooldown LR at min_lr, after cyclic schedule ends",
    )
    parser.add_argument(
        "--patience-epochs",
        type=int,
        default=10,
        metavar="N",
        help="patience epochs for Plateau LR scheduler (default: 10",
    )
    parser.add_argument(
        "--decay-rate",
        "--dr",
        type=float,
        default=0.1,
        metavar="RATE",
        help="LR decay rate (default: 0.1)",
    )

    # Augmentation & regularization parameters
    parser.add_argument(
        "--no-aug",
        action="store_true",
        default=False,
        help="Disable all training augmentation, override other train aug args",
    )
    parser.add_argument(
        "--scale",
        type=float,
        nargs="+",
        default=[0.08, 1.0],
        metavar="PCT",
        help="Random resize scale (default: 0.08 1.0)",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        nargs="+",
        default=[3.0 / 4.0, 4.0 / 3.0],
        metavar="RATIO",
        help="Random resize aspect ratio (default: 0.75 1.33)",
    )
    parser.add_argument(
        "--hflip",
        type=float,
        default=0.5,
        help="Horizontal flip training aug probability",
    )
    parser.add_argument(
        "--vflip",
        type=float,
        default=0.0,
        help="Vertical flip training aug probability",
    )
    parser.add_argument(
        "--color-jitter",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Color jitter factor (default: 0.4)",
    )
    parser.add_argument(
        "--aa",
        type=str,
        default=None,
        metavar="NAME",
        help='Use AutoAugment policy. "v0" or "original". (default: None)',
    ),
    parser.add_argument(
        "--aug-splits",
        type=int,
        default=0,
        help="Number of augmentation splits (default: 0, valid: 0 or >=2)",
    )
    parser.add_argument(
        "--jsd",
        action="store_true",
        default=False,
        help="Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.",
    )
    parser.add_argument(
        "--reprob",
        type=float,
        default=0.0,
        metavar="PCT",
        help="Random erase prob (default: 0.)",
    )
    parser.add_argument(
        "--remode",
        type=str,
        default="const",
        help='Random erase mode (default: "const")',
    )
    parser.add_argument(
        "--recount", type=int, default=1, help="Random erase count (default: 1)"
    )
    parser.add_argument(
        "--resplit",
        action="store_true",
        default=False,
        help="Do not random erase first (clean) augmentation split",
    )
    parser.add_argument(
        "--smoothing", type=float, default=0.0, help="Label smoothing (default: 0.0)"
    )
    parser.add_argument(
        "--smoothed_l1", default=False, action="store_true", help="L1 smooth"
    )
    parser.add_argument(
        "--train-interpolation",
        type=str,
        default="random",
        help='Training interpolation (random, bilinear, bicubic default: "random")',
    )
    parser.add_argument(
        "--drop",
        type=float,
        default=0.0,
        metavar="PCT",
        help="Dropout rate (default: 0.)",
    )
    parser.add_argument(
        "--drop-connect",
        type=float,
        default=None,
        metavar="PCT",
        help="Drop connect rate, DEPRECATED, use drop-path (default: None)",
    )
    parser.add_argument(
        "--drop-path",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Drop path rate (default: None)",
    )
    parser.add_argument(
        "--drop-block",
        type=float,
        default=None,
        metavar="PCT",
        help="Drop block rate (default: None)",
    )

    # Batch norm parameters (only works with gen_efficientnet based models currently)
    parser.add_argument(
        "--bn-tf",
        action="store_true",
        default=False,
        help="Use Tensorflow BatchNorm defaults for models that support it (default: False)",
    )
    parser.add_argument(
        "--bn-momentum",
        type=float,
        default=None,
        help="BatchNorm momentum override (if not None)",
    )
    parser.add_argument(
        "--bn-eps",
        type=float,
        default=None,
        help="BatchNorm epsilon override (if not None)",
    )
    parser.add_argument(
        "--sync-bn",
        action="store_true",
        help="Enable NVIDIA Apex or Torch synchronized BatchNorm.",
    )
    parser.add_argument(
        "--dist-bn",
        type=str,
        default="",
        help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")',
    )
    parser.add_argument(
        "--split-bn",
        action="store_true",
        help="Enable separate BN layers per augmentation split.",
    )

    # Model Exponential Moving Average
    parser.add_argument(
        "--model-ema",
        action="store_true",
        default=False,
        help="Enable tracking moving average of model weights",
    )
    parser.add_argument(
        "--model-ema-force-cpu",
        action="store_true",
        default=False,
        help="Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.",
    )
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.9998,
        help="decay factor for model weights moving average (default: 0.9998)",
    )

    # Misc
    parser.add_argument(
        "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--recovery-interval",
        type=int,
        default=0,
        metavar="N",
        help="how many batches to wait before writing recovery checkpoint",
    )
    parser.add_argument(
        "--checkpoint-hist",
        type=int,
        default=5,
        metavar="N",
        help="number of checkpoints to keep (default: 10)",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help="how many training processes to use (default: 1)",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        default=False,
        help="save images of input bathes every log interval for debugging",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=False,
        help="use NVIDIA Apex AMP or Native AMP for mixed precision training",
    )
    parser.add_argument(
        "--apex-amp",
        action="store_true",
        default=False,
        help="Use NVIDIA Apex AMP mixed precision",
    )
    parser.add_argument(
        "--native-amp",
        action="store_true",
        default=False,
        help="Use Native Torch AMP mixed precision",
    )
    parser.add_argument(
        "--channels-last",
        action="store_true",
        default=False,
        help="Use channels_last memory layout",
    )
    parser.add_argument(
        "--pin-mem",
        action="store_true",
        default=False,
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument(
        "--no-prefetcher",
        action="store_true",
        default=False,
        help="disable fast prefetcher",
    )
    parser.add_argument(
        "--output",
        type=str,
        metavar="PATH",
        help="path to output folder (default: none, current dir)",
    )
    parser.add_argument(
        "--experiment",
        default="",
        type=str,
        metavar="NAME",
        help="name of train experiment, name of sub-folder for output",
    )
    parser.add_argument(
        "--eval-metric",
        default="top1",
        type=str,
        metavar="EVAL_METRIC",
        help='Best metric (default: "top1"',
    )
    parser.add_argument(
        "--tta",
        type=int,
        default=0,
        metavar="N",
        help="Test/inference time augmentation (oversampling) factor. 0=None (default: 0)",
    )
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument(
        "--use-multi-epochs-loader",
        action="store_true",
        default=False,
        help="use the multi-epochs-loader to save time at the beginning of every epoch",
    )
    parser.add_argument(
        "--torchscript",
        dest="torchscript",
        action="store_true",
        help="convert model torchscript for inference",
    )
    parser.add_argument(
        "--log-wandb",
        action="store_true",
        default=False,
        help="log training and validation metrics to wandb",
    )

    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    return args, args_text
