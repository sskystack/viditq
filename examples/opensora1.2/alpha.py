from opensora.models.stdit.stdit3 import STDiT3Config
from models.quant_opensora import QuantOpenSora
from omegaconf import OmegaConf,ListConfig
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
    progress_wrap = tqdm if verbose == 1 else (lambda x: x)
    
    # INFO: precompute the text embeds to avoid loading the T5 repeatedly
    precompute_text_embeds = cfg.get("precompute_text_embeds", False)
    assert precompute_text_embeds # DEBUG_ONLY

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
    model_from_pretrained="/home/zhaotianchen/models/hpcai-tech/OpenSora-STDiT-v3"  # DIRTY
    

    # == build scheduler ==
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    ALPHA=[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
    for id_a in range(len(ALPHA)):
        quant_config.viditq.alpha=ALPHA[id_a]/10+0.545
        model=(QuantOpenSora(quant_config,config,model_from_pretrained).to(device, dtype).eval())  
        if not precompute_text_embeds:
            text_encoder.y_embedder = model.y_embedder  # HACK: for classifier-free guidance
        if_mixed_precision = isinstance(quant_config.weight.n_bits, ListConfig) or isinstance(quant_config.act.n_bits, ListConfig)
        if if_mixed_precision:
            model.bitwidth_refactor()
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
            
        model.set_init_done()
        # layer = model.spatial_blocks[8]  
        # weights = layer.state_dict()["attn.qkv.weight"]
        # weight_name=str(id_a)
        # torch.save(weights, f'{weight_name}.pth')
        logger.info(str(model))

        # ======================================================
        # inference
        # ======================================================
        # == load prompts ==
        prompts = cfg.get("prompt", None)
        start_idx = cfg.get("start_index", 0)
        if prompts is None:
            if cfg.get("prompt_path", None) is not None:
                prompts = load_prompts(cfg.prompt_path, start_idx, cfg.get("end_index", None))
            else:
                prompts = [cfg.get("prompt_generator", "")] * 1_000_000  # endless loop
        
        # == prepare reference ==
        reference_path = cfg.get("reference_path", [""] * len(prompts))
        mask_strategy = cfg.get("mask_strategy", [""] * len(prompts))
        assert len(reference_path) == len(prompts), "Length of reference must be the same as prompts"
        assert len(mask_strategy) == len(prompts), "Length of mask_strategy must be the same as prompts"

        # == prepare arguments ==
        fps = cfg.fps
        save_fps = cfg.get("save_fps", fps // cfg.get("frame_interval", 1))
        multi_resolution = cfg.get("multi_resolution", None)
        batch_size = cfg.get("batch_size", 1)
        num_sample = cfg.get("num_sample", 1)
        loop = cfg.get("loop", 1)
        condition_frame_length = cfg.get("condition_frame_length", 5)
        condition_frame_edit = cfg.get("condition_frame_edit", 0.0)
        align = cfg.get("align", None)

        save_dir = cfg.save_dir
        os.makedirs(save_dir, exist_ok=True)
        sample_name = cfg.get("sample_name", None)
        prompt_as_path = cfg.get("prompt_as_path", False)

            # == Iter over all samples ==
        for i in progress_wrap(range(0, len(prompts), batch_size)):
            # == prepare batch prompts ==
            batch_prompts = prompts[i : i + batch_size]
            ms = mask_strategy[i : i + batch_size]
            refs = reference_path[i : i + batch_size]

            # == get json from prompts ==
            batch_prompts, refs, ms = extract_json_from_prompts(batch_prompts, refs, ms)
            original_batch_prompts = batch_prompts

            # == get reference for condition ==
            refs = collect_references_batch(refs, vae, image_size)

            # == multi-resolution info ==
            model_args = prepare_multi_resolution_info(
                multi_resolution, len(batch_prompts), image_size, num_frames, fps, device, dtype
            )

            # == Iter over number of sampling for one prompt ==
            for k in range(num_sample):
                # == prepare save paths ==
                save_paths = [
                    get_save_path_name(
                        save_dir,
                        sample_name=sample_name,
                        sample_idx=id_a,
                        prompt=original_batch_prompts[idx],
                        prompt_as_path=prompt_as_path,
                        num_sample=num_sample,
                        k=k,
                    )
                    for idx in range(len(batch_prompts))
                ]

                # NOTE: Skip if the sample already exists
                # This is useful for resuming sampling VBench
                if prompt_as_path and all_exists(save_paths):
                    continue

                # == process prompts step by step ==
                # 0. split prompt
                # each element in the list is [prompt_segment_list, loop_idx_list]
                batched_prompt_segment_list = []
                batched_loop_idx_list = []
                for prompt in batch_prompts:
                    prompt_segment_list, loop_idx_list = split_prompt(prompt)
                    batched_prompt_segment_list.append(prompt_segment_list)
                    batched_loop_idx_list.append(loop_idx_list)

                # 1. refine prompt by openai
                if cfg.get("llm_refine", False):
                    # only call openai API when
                    # 1. seq parallel is not enabled
                    # 2. seq parallel is enabled and the process is rank 0
                    if not enable_sequence_parallelism or (enable_sequence_parallelism and is_main_process()):
                        for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                            batched_prompt_segment_list[idx] = refine_prompts_by_openai(prompt_segment_list)

                    # sync the prompt if using seq parallel
                    if enable_sequence_parallelism:
                        coordinator.block_all()
                        prompt_segment_length = [
                            len(prompt_segment_list) for prompt_segment_list in batched_prompt_segment_list
                        ]

                        # flatten the prompt segment list
                        batched_prompt_segment_list = [
                            prompt_segment
                            for prompt_segment_list in batched_prompt_segment_list
                            for prompt_segment in prompt_segment_list
                        ]

                        # create a list of size equal to world size
                        broadcast_obj_list = [batched_prompt_segment_list] * coordinator.world_size
                        dist.broadcast_object_list(broadcast_obj_list, 0)

                        # recover the prompt list
                        batched_prompt_segment_list = []
                        segment_start_idx = 0
                        all_prompts = broadcast_obj_list[0]
                        for num_segment in prompt_segment_length:
                            batched_prompt_segment_list.append(
                                all_prompts[segment_start_idx : segment_start_idx + num_segment]
                            )
                            segment_start_idx += num_segment

                # 2. append score
                for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                    batched_prompt_segment_list[idx] = append_score_to_prompts(
                        prompt_segment_list,
                        aes=cfg.get("aes", None),
                        flow=cfg.get("flow", None),
                        camera_motion=cfg.get("camera_motion", None),
                    )

                # 3. clean prompt with T5
                for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                    batched_prompt_segment_list[idx] = [text_preprocessing(prompt) for prompt in prompt_segment_list]

                # 4. merge to obtain the final prompt
                batch_prompts = []
                for prompt_segment_list, loop_idx_list in zip(batched_prompt_segment_list, batched_loop_idx_list):
                    batch_prompts.append(merge_prompt(prompt_segment_list, loop_idx_list))

                # == Iter over loop generation ==
                video_clips = []
                for loop_i in range(loop):
                    # == get prompt for loop i ==
                    batch_prompts_loop = extract_prompts_loop(batch_prompts, loop_i)

                    # == add condition frames for loop ==
                    if loop_i > 0:
                        refs, ms = append_generated(
                            vae, video_clips[-1], refs, ms, loop_i, condition_frame_length, condition_frame_edit
                        )

                    # == sampling ==
                    seed_everything(cfg.get("seed", 1024))  # DEBUG: somehow even if we have already set the seed before, we need to reinit here, maybe the seed changes within some imported files?
                    z = torch.randn(len(batch_prompts), vae.out_channels, *latent_size, device=device, dtype=dtype)
                    masks = apply_mask_strategy(z, refs, ms, loop_i, align=align)
                    samples = scheduler.sample(
                        model,
                        text_encoder if not precompute_text_embeds else None,
                        z=z,
                        prompts=batch_prompts_loop,
                        device=device,
                        additional_args=model_args,
                        progress=verbose >= 2,
                        mask=masks,
                        precompute_text_embeds=precompute_text_embeds,
                    )
                    samples = vae.decode(samples.to(dtype), num_frames=num_frames)
                    video_clips.append(samples)

                # == save samples ==
                if is_main_process():
                    for idx, batch_prompt in enumerate(batch_prompts):
                        if verbose >= 2:
                            logger.info("Prompt: %s", batch_prompt)
                        save_path = save_paths[idx]
                        video = [video_clips[i][idx] for i in range(loop)]
                        for i in range(1, loop):
                            video[i] = video[i][:, dframe_to_frame(condition_frame_length) :]
                        video = torch.cat(video, dim=1)
                        save_path = save_sample(
                            video,
                            fps=save_fps,
                            save_path=save_path,
                            verbose=verbose >= 2,
                        )
                        if save_path.endswith(".mp4") and cfg.get("watermark", False):
                            time.sleep(1)  # prevent loading previous generated video
                            add_watermark(save_path)
            start_idx += len(batch_prompts)
        logger.info("Inference finished.")
        logger.info("Saved %s samples to %s", start_idx, save_dir)

if __name__ == "__main__":
    main()

