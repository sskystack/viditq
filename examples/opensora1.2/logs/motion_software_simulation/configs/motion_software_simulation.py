resolution = "512"
aspect_ratio = "1:1"
num_frames = 64
fps = 24
frame_interval = 1
save_fps = 24
ptq_config = "./configs/motion_w4a48.yaml"
save_dir = "./logs/motion_software_simulation"
seed = 42
batch_size = 1
multi_resolution = "STDiT2"
dtype = "fp16"
condition_frame_length = 5
align = 5

model = dict(
    type="STDiT3-XL/2",
    from_pretrained="/home/aistudio/data/models/hpcai-tech/OpenSora-STDiT-v3",
    qk_norm=True,
    enable_flash_attn=False,
    enable_layernorm_kernel=False,
    force_huggingface=True,
)

vae = dict(
    type="OpenSoraVAE_V1_2",
    from_pretrained="/home/aistudio/data/models/hpcai-tech/OpenSora-VAE-v1.2",
    micro_frame_size=17,
    micro_batch_size=4,
    force_huggingface=True,
)

text_encoder = dict(
    type="t5",
    from_pretrained="/home/aistudio/data/models/DeepFloyd/t5-v1_1-xxl",
    model_max_length=300,
)

scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    num_sampling_steps=30,
    cfg_scale=7.0,
)

aes = 6.5
flow = None

precompute_text_embeds = False
prompt_path = "./assets/t2v_samples.txt"
model_path = "/home/aistudio/data/models"
hardware = False

