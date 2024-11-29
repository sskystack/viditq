# iter through some folder to test all metrics
target_dir="/mnt/public/diffusion_quant/zhaotianchen/project/attn_quant/diffuser-dev/examples/cogvideo_attn/logs/baselines_50steps"

for folder in $(find "$target_dir" -type d -maxdepth 1 -mindepth 1 -exec basename {} \;); do
    # 将文件夹名称作为 $1 参数传入 main.sh

    echo "$target_dir/$folder/generated_videos"
    ./simplebench_comp_101.sh  "$target_dir/$folder/generated_videos"

done


