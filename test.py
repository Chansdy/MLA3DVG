import math
import datetime
from pathlib import Path

from tqdm.auto import tqdm

from utils.logging import get_file_handler
from utils.parser import parse_args, load_config
from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.accelerator_helper import build_accelerator, init_accelerator, get_mp_logger as get_logger
from lib.helpers.checkpoint_helper import CustomCheckpoint, get_resume_chekpoint_path
from lib.helpers.metric_helper import evaluation
from lib.models.configuration_mla3dvg import MLA3DVGConfig
from lib.models.mla3dvg import Mono3DVGForSingleObjectDetection as Mono3DVG
from lib.models.image_processsing_mono3dvg import Mono3DVGImageProcessor
from mmcv.runner import load_checkpoint
import depth_estimation
import torch
import mmcv
logger = get_logger(__name__)


def main():
    args = parse_args()
    cfg = load_config(args, args.cfg_file)
    cfg.with_tracking = False
    
    accelerator = build_accelerator(cfg)
    # Handle the hugingface hub repo creation
    if accelerator.is_main_process:
        if cfg.output_dir is not None:
            Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)    
    accelerator.wait_for_everyone()
    
    #logger.logger.addHandler(get_file_handler(Path(cfg.output_dir) / f'test.{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log'))
    
    logger.info(f'Init accelerator...\n{accelerator.state}', main_process_only=False)
    
    logger.info("Init DataLoader...")
    _, _, test_dataloader, id2label, label2id = build_dataloader(
        cfg, workers=cfg.dataloader_num_workers, accelerator=accelerator
    )
    
    logger.info("Init Model...")
    config = MLA3DVGConfig(
        label2id=label2id, id2label=id2label, **vars(cfg.model)
    )
    model = Mono3DVG(config)
    image_processor = Mono3DVGImageProcessor()
    
    BACKBONE_SIZE = "small" # in ("small", "base", "large" or "giant")
    backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }
    backbone_arch = backbone_archs[BACKBONE_SIZE]
    backbone_name = f"dinov2_{backbone_arch}"
    backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name).cuda()
    backbone_model.eval()
    HEAD_DATASET = "kitti" # in ("nyu", "kitti")
    HEAD_TYPE = "dpt" # in ("linear", "linear4", "dpt")
    DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
    head_config_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_config.py"
    head_checkpoint_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_head.pth"

    # cfg_str = depth_estimation.load_config_from_url(head_config_url)
    # cfg_d = mmcv.Config.fromstring(cfg_str, file_format=".py")
    # cfg_d = mmcv.Config.fromfile("/data/csd/Mono3DVGTR-main/pretrained-models/dinov2_vitl14_kitti_dpt_config.py")
    cfg_d = mmcv.Config.fromfile("./pretrained-models/dinov2_vits14_kitti_dpt_config.py")

    model_depth = depth_estimation.create_depther(
        cfg_d,
        backbone_model=backbone_model,
        backbone_size=BACKBONE_SIZE,
        head_type=HEAD_TYPE,
    )
    
    load_checkpoint(model_depth, head_checkpoint_url, map_location="cpu")
    model_depth = model_depth.to('cuda')
    model_depth.eval() 
    # Prepare everything with our `accelerator`.
    model, test_dataloader = accelerator.prepare(
        model, test_dataloader
    )
    
    # We need to recalculate our total test steps as the size of the testing dataloader may have changed.
    cfg.max_test_steps = math.ceil(len(test_dataloader) / cfg.gradient_accumulation_steps)
    
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    init_accelerator(accelerator, cfg)
    
    # ------------------------------------------------------------------------------------------------
    # Run testing
    # ------------------------------------------------------------------------------------------------

    total_batch_size = cfg.test_batch_size * accelerator.num_processes * cfg.gradient_accumulation_steps

    logger.info("##################  Running testing  ##################")
    logger.info(f"  Num examples = {len(test_dataloader.dataset)}")
    logger.info(f"  Instantaneous batch size per device = {cfg.test_batch_size}")
    logger.info(f"  Total test batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.gradient_accumulation_steps}")
    logger.info(f"  Total testing steps = {cfg.max_test_steps}")

    # custom checkpoint data registe to accelerator
    extra_state = CustomCheckpoint()
    
    # Potentially load in the weights and states from a previous save
    if cfg.pretrain_model:
        checkpoint_path = get_resume_chekpoint_path(cfg.pretrain_model, cfg.output_dir)
        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.register_for_checkpointing(extra_state)
        accelerator.load_state(checkpoint_path)
        logger.info(f"Loading Checkpoint... Best Result:{extra_state.best_result}, Best Epoch:{extra_state.best_epoch}")
    
    logger.info("***** Running evaluation *****")
    
    metrics = evaluation(model, model_depth,image_processor, accelerator, test_dataloader, 
                         logger=logger, only_overall=False)
    for split_name in ['Overall', 'Unique', 'Multiple', 'Near', 'Medium', 'Far', 'Easy', 'Moderate', 'Hard']:
        logger.info(f"------------{split_name}------------")
        msg = (
            f'Accuracy@0.25: {metrics[f"{split_name}_Acc@0.25"]:.2f}%\t'
            f'Accuracy@0.5: {metrics[f"{split_name}_Acc@0.5"]:.2f}%\t'
            f'Mean IoU: {metrics[f"{split_name}_MeanIoU"]:.2f}%\t'
        )
        logger.info(msg)


if __name__ == '__main__':
    main()
