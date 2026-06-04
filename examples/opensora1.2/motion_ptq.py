from opensora.models.stdit.stdit3 import STDiT3Config
from models.quant_opensora_motion import QuantOpenSoraMotion
from omegaconf import OmegaConf, ListConfig
import os
import time
from pprint import pformat

import colossalai
import torch
import torch.distributed as dist
from colossalai.cluster import DistCoordinator
from mmengine.runner import set_random_seed
from tqdm import tqdm

from opensora.acceleration.parallel_states import set_sequence_parallel_group
from opensora.datasets import save_sample
from opensora.datasets.aspect import get_image_size, get_num_frames
from opensora.models.text_encoder.t5 import text_preprocessing
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.inference_utils import (
    add_watermark,
    append_generated,
    append_score_to_prompts,
    apply_mask_strategy,
    collect_references_batch,
    dframe_to_frame,
    extract_json_from_prompts,
    extract_prompts_loop,
    get_save_path_name,
    load_prompts,
    merge_prompt,
    prepare_multi_resolution_info,
    refine_prompts_by_openai,
    split_prompt,
)
from opensora.utils.misc import all_exists, create_logger, is_distributed, is_main_process, to_torch_dtype
from qdiff.utils import apply_func_to_submodules, seed_everything
def main():
    torch.set_grad_enabled(False)
    # ======================================================
    # configs & runtime variables
    # ======================================================
    # == parse configs ==
    cfg = parse_configs(training=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_dtype = cfg.get("dtype", "fp32")
    assert cfg_dtype in ["fp16", "bf16", "fp32"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # == init distributed env ==
    if is_distributed():
        colossalai.launch_from_torch({})
        coordinator = DistCoordinator()
        enable_sequence_parallelism = coordinator.world_size > 1
        if enable_sequence_parallelism:
            set_sequence_parallel_group(dist.group.WORLD)
    else:
        coordinator = None
        enable_sequence_parallelism = False
    seed_everything(cfg.get("seed", 1024))
    # set_random_seed(seed=cfg.get("seed", 1024))
    
    # == bakup some files ==
    import shutil
    if os.path.exists(os.path.join(cfg.save_dir,'configs')):
        shutil.rmtree(os.path.join(cfg.save_dir,'configs'))
    shutil.copytree('./configs', os.path.join(cfg.save_dir,'configs'))

    # == init logger ==
    logger = create_logger()
    logger.info("Inference configuration:\n %s", pformat(cfg.to_dict()))
    verbose = cfg.get("verbose", 1)
    def progress_wrap(iterable, desc=None, total=None):
        if verbose == 1:
            return tqdm(iterable, desc=desc, total=total, dynamic_ncols=True)
        return iterable
    calib_data = None
    
    # INFO: precompute the text embeds to avoid loading the T5 repeatedly
    precompute_text_embeds = cfg.get("precompute_text_embeds", False)
    #assert precompute_text_embeds # DEBUG_ONLY

    # ======================================================
    # build model & load weights
    # ======================================================
    logger.info("Building models...")
    # == build text-encoder and vae ==
    if not precompute_text_embeds:
        text_encoder = build_module(cfg.text_encoder, MODELS, device=device)
    vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()


    # == prepare video size ==
    image_size = cfg.get("image_size", None)
    if image_size is None:
        resolution = cfg.get("resolution", None)
        aspect_ratio = cfg.get("aspect_ratio", None)
        assert (
            resolution is not None and aspect_ratio is not None
        ), "resolution and aspect_ratio must be provided if image_size is not provided"
        image_size = get_image_size(resolution, aspect_ratio)
    num_frames = get_num_frames(cfg.num_frames)

    # == build diffusion model ==
    quant_config = cfg.get("ptq_config", None)
    quant_config = OmegaConf.load(quant_config)
    input_size = (num_frames, *image_size)
    latent_size = vae.get_latent_size(input_size)
    config = STDiT3Config(depth=28, 
                        hidden_size=1152, 
                        patch_size=(1, 2, 2), 
                        num_heads=16, 
                        qk_norm=True,
                        enable_flash_attn=False,
                        enable_layernorm_kernel=False,  # no apex included
                        input_size=latent_size,
                        in_channels=vae.out_channels,
                        caption_channels=text_encoder.output_dim if not precompute_text_embeds else 4096,
                        model_max_length=text_encoder.model_max_length if not precompute_text_embeds else 300,
                        enable_sequence_parallelism=enable_sequence_parallelism)
    model_from_pretrained=os.path.join(cfg.model_path, "hpcai-tech/OpenSora-STDiT-v3")
    model=(QuantOpenSoraMotion(quant_config,config,model_from_pretrained).to(device, dtype).eval())  
    if not precompute_text_embeds:
        text_encoder.y_embedder = model.y_embedder  # HACK: for classifier-free guidance
    if_mixed_precision = isinstance(quant_config.weight.n_bits, ListConfig) or isinstance(quant_config.act.n_bits, ListConfig)
    if if_mixed_precision:
        model.bitwidth_refactor()
    # == build scheduler ==
    scheduler = build_module(cfg.scheduler, SCHEDULERS)
    
    '''
    INFO: The PTQ process:
    for simple PTQ with dynamic act quant: 
    the weight are quantized with quant_model initialization.
    the act quant params are calculated online. 
    '''
    
    # TODO: some variables (quant_config) are not replaced yet, havent test sq and quarot!
    
    def init_sq_channel_mask_(module, full_name, calib_data):
        assert isinstance(module, SQQuantizedLinear)
        act_mask = calib_data[full_name].max(dim=0)[0]  # [T, C], averaged over all timesteps
        zero_mask = act_mask < 1e-3
        act_mask = torch.where(zero_mask, torch.tensor(1e-3), act_mask)
        module.get_channel_mask(act_mask)  # set self.channel_mask
        module.update_quantized_weight_scaled()

    def init_rotation_matrix_(module, full_name):
        from qdiff.quarot.quarot_utils import random_hadamard_matrix, matmul_hadU_cuda
        assert isinstance(module, QuarotQuantizedLinear)
        module.get_rotation_matrix()
        module.update_quantized_weight_rotated()
    
    def init_rotation_and_channel_mask_(module, full_name, calib_data):
        assert isinstance(module, ViDiTQuantizedLinear)
        act_mask = calib_data[full_name].max(dim=0)[0]  # [T, C], averaged over all timesteps
        zero_mask = act_mask < 1e-3
        act_mask = torch.where(zero_mask, torch.tensor(1e-3), act_mask)
        module.get_channel_mask(act_mask)  # set self.channel_mask
        module.get_rotation_matrix()
        module.update_quantized_weight_rotated_and_scaled()

    '''
    INFO: the smooth_quant quantization.
    load act channel mask from the calib data
    '''
    if quant_config.get("smooth_quant",None) is not None:
        # INFO: the SQQuantizedLayer are initialized with the quant_layer_refactor_ in quant_dit.py
        from qdiff.smooth_quant.sq_quant_layer import SQQuantizedLinear

        assert quant_config.calib_data.save_path is not None
        calib_data = torch.load(quant_config.calib_data.save_path, weights_only=True)  # default wtih weights_only=True, will cause warning

        # get the channel mask, iter through all layers
        kwargs = {}
        apply_func_to_submodules(model,
                            class_type=SQQuantizedLinear,  # add hook to all objects of this cls
                            function=init_sq_channel_mask_,
                            calib_data = calib_data,
                            full_name='',
                            **kwargs
                            )

    '''
    INFO: the quarot quantization.
    init and apply the rotation matrix
    '''
    if quant_config.get("quarot",None) is not None:
        
        from qdiff.quarot.quarot_quant_layer import QuarotQuantizedLinear
        # get the rotation matrix, iter through all layers
        kwargs = {}
        apply_func_to_submodules(model,
                            class_type=QuarotQuantizedLinear,  # add hook to all objects of this cls
                            function=init_rotation_matrix_,
                            full_name='',
                            **kwargs
                            )
    '''
    INFO: combining both
    '''
    if quant_config.get("viditq",None) is not None:
        from qdiff.viditq.viditq_quant_layer import ViDiTQuantizedLinear
        
        assert quant_config.calib_data.save_path is not None
        calib_data = torch.load(quant_config.calib_data.save_path, weights_only=True)  # default wtih 
        kwargs = {}
        apply_func_to_submodules(model,
                            class_type=ViDiTQuantizedLinear,  # add hook to all objects of this cls
                            function=init_rotation_and_channel_mask_,
                            full_name='',
                            calib_data = calib_data,
                            **kwargs
                            )

    def build_motion_clip_from_calib_data():
        motion_cfg = quant_config.get("motion_ptq", None)
        if motion_cfg is None or not motion_cfg.get("enabled", False):
            return {}
        if calib_data is None:
            logger.warning("motion clip calibration requested from calib_data, but calib_data is not loaded.")
            return {}

        from qdiff.opensora_motion.motion_quant_layer import MotionQuantizedLinear

        clip_quantile = float(motion_cfg.get("clip_quantile", 0.999))

        def build_layer_clip_(module, full_name):
            if full_name not in calib_data:
                return {}
            act_stat = calib_data[full_name].detach().float().abs().reshape(-1)
            if act_stat.numel() == 0:
                return {}
            clip = torch.quantile(act_stat, clip_quantile).detach().cpu()
            return {
                str(module.low_bit): clip,
                str(module.mid_bit): clip,
                str(module.high_bit): clip,
            }

        clip_dict = apply_func_to_submodules(
            model,
            class_type=MotionQuantizedLinear,
            function=build_layer_clip_,
            return_d={},
            full_name=None,
        )
        clip_dict = {name: clips for name, clips in clip_dict.items() if len(clips) > 0}
        logger.info(f"Built motion clip params from calib_data for {len(clip_dict)} motion-quantized layer(s).")
        return clip_dict

    def move_probe_arg(value):
        if isinstance(value, torch.Tensor):
            if torch.is_floating_point(value):
                return value.to(device=device, dtype=dtype)
            return value.to(device=device)
        return value

    def replay_motion_probe_calibration():
        motion_cfg = quant_config.get("motion_ptq", None)
        if motion_cfg is None or not motion_cfg.get("enabled", False):
            return {}

        probe_cache_path = motion_cfg.get("probe_cache_path", "./motion_probe_cache.pth")
        if not os.path.exists(probe_cache_path):
            raise FileNotFoundError(
                f"motion_ptq.clip_calib_mode='probe_replay' requires latent probe cache: {probe_cache_path}. "
                "Run get_calib_data.py first with the same config."
            )

        probe_cache = torch.load(probe_cache_path, map_location="cpu", weights_only=False)
        records = probe_cache.get("records", [])
        if len(records) == 0:
            raise ValueError(f"No motion latent probes found in {probe_cache_path}.")

        logger.info(
            "Replaying %s motion latent probe(s) from %s for bucket-wise activation clips.",
            len(records),
            probe_cache_path,
        )
        model.enable_motion_calibration(enabled=True, reset=True, observe_only=True)
        model.set_init_done()

        replay_iter = tqdm(records, desc="motion probe replay", dynamic_ncols=True) if verbose >= 1 else records
        with torch.no_grad():
            for record_idx, record in enumerate(replay_iter):
                z = record["latent"].to(device=device, dtype=dtype)
                timestep = record["timestep"].to(device=device, dtype=dtype)
                model_args = {key: move_probe_arg(value) for key, value in record["model_args"].items()}
                z_in = torch.cat([z, z], 0)
                t_in = torch.cat([timestep, timestep], 0)
                _ = model(z_in, t_in, **model_args)
                if verbose >= 1 and hasattr(replay_iter, "set_postfix"):
                    replay_iter.set_postfix(
                        step=record.get("step_index", "?"),
                        done=f"{record_idx + 1}/{len(records)}",
                    )

        model.enable_motion_calibration(enabled=False, reset=False, observe_only=False)
        motion_clip_dict = model.get_motion_clip_dict()
        logger.info(f"Replayed motion probes and collected clip stats for {len(motion_clip_dict)} layer(s).")
        return motion_clip_dict

    def run_motion_clip_calibration():
        motion_cfg = quant_config.get("motion_ptq", None)
        if motion_cfg is None or not motion_cfg.get("enabled", False):
            return {}

        clip_calib_mode = motion_cfg.get("clip_calib_mode", "act_calib")
        if clip_calib_mode == "probe_replay":
            return replay_motion_probe_calibration()
        if clip_calib_mode == "act_calib":
            return build_motion_clip_from_calib_data()
        if clip_calib_mode == "none":
            logger.info("Skipping motion clip calibration by config.")
            return {}
        if clip_calib_mode not in ("observe_only", "sample"):
            raise ValueError(f"Unsupported motion_ptq.clip_calib_mode: {clip_calib_mode}")

        assert not precompute_text_embeds, "Motion PTQ calibration currently expects an online text encoder."

        prompts = cfg.get("prompt", None)
        if prompts is None:
            if cfg.get("prompt_path", None) is not None:
                prompts = load_prompts(cfg.prompt_path, cfg.get("start_index", 0), cfg.get("end_index", None))
            else:
                prompts = [cfg.get("prompt_generator", "")]
        calib_num_samples = int(motion_cfg.get("calib_num_samples", 0))
        if calib_num_samples > 0:
            prompts = prompts[:calib_num_samples]
        observe_only = clip_calib_mode == "observe_only"
        logger.info(
            f"Collecting motion bucket activation clips with {len(prompts)} prompt(s) "
            f"(mode={clip_calib_mode})."
        )

        model.enable_motion_calibration(enabled=True, reset=True, observe_only=observe_only)
        model.set_init_done()

        batch_size = cfg.get("batch_size", 1)
        align = cfg.get("align", None)
        reference_path = cfg.get("reference_path", [""] * len(prompts))
        mask_strategy = cfg.get("mask_strategy", [""] * len(prompts))

        with torch.no_grad():
            total_batches = (len(prompts) + batch_size - 1) // batch_size
            for i in progress_wrap(range(0, len(prompts), batch_size), desc="motion clip prompts", total=total_batches):
                batch_prompts = prompts[i : i + batch_size]
                refs = reference_path[i : i + batch_size]
                ms = mask_strategy[i : i + batch_size]
                batch_prompts, refs, ms = extract_json_from_prompts(batch_prompts, refs, ms)
                refs = collect_references_batch(refs, vae, image_size)
                batch_prompts = [text_preprocessing(prompt) for prompt in batch_prompts]
                model_args = prepare_multi_resolution_info(
                    cfg.get("multi_resolution", None), len(batch_prompts), image_size, num_frames, cfg.fps, device, dtype
                )

                seed_everything(cfg.get("seed", 1024))
                z = torch.randn(len(batch_prompts), vae.out_channels, *latent_size, device=device, dtype=dtype)
                masks = apply_mask_strategy(z, refs, ms, 0, align=align)
                _ = scheduler.sample(
                    model,
                    text_encoder,
                    z=z,
                    prompts=batch_prompts,
                    device=device,
                    additional_args=model_args,
                    progress=verbose >= 1,
                    mask=masks,
                    precompute_text_embeds=False,
                )

        model.enable_motion_calibration(enabled=False, reset=False, observe_only=False)
        motion_clip_dict = model.get_motion_clip_dict()
        logger.info(f"Collected motion clip stats for {len(motion_clip_dict)} motion-quantized layer(s).")
        return motion_clip_dict

    motion_clip_dict = run_motion_clip_calibration()
    if len(motion_clip_dict) > 0:
        torch.save(motion_clip_dict, os.path.join(cfg.save_dir, "motion_clip_params.pth"))
        logger.info(f"saved motion clip params into {cfg.save_dir}")

    model.set_init_done()
    model.save_quant_param_dict()
    torch.save(model.quant_param_dict, os.path.join(cfg.save_dir, 'motion_quant_params.pth'))
    logger.info(f'saved quant params into {cfg.save_dir}')

if __name__ == "__main__":
    main()
