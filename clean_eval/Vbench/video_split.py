import os
import shutil

# 定义文件夹路径
folder_a = '/mnt/public/fangtongcheng/iclr_code/ViDiT-Q/logs/exp_iclr/qdit_w6a6_test/generated_videos_opensora'  # 文件夹A的路径
folder_b = '/mnt/public/fangtongcheng/iclr_code/ViDiT-Q/logs/exp_iclr/qdit_w6a6_test/processed_videos'  # 文件夹B的路径
folder_txt = '/share/public/video_quant/wanrui/VBench/final_prompt_simple'
# 定义子文件夹路径
subfolders = ['subject_consistency', 'overall_consistency', 'scene']
subfolder_paths = [os.path.join(folder_b, subfolder) for subfolder in subfolders]

# 确保目标文件夹及其子文件夹存在，不存在则创建
if not os.path.exists(folder_b):
    os.makedirs(folder_b)

for subfolder_path in subfolder_paths:
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)

# 获取文件夹A中的所有文件，并按名称排序
files = sorted([f for f in os.listdir(folder_a) if f.startswith('sample_') and f.endswith('.mp4')],
               key=lambda x: int(x.split('_')[1].split('.')[0]))

# 定义分区点
cut1 = 72
cut2 = 72 + 93

# 移动文件到相应的子文件夹
for i, file in enumerate(files):
    src_path = os.path.join(folder_a, file)
    if i < cut1:
        dest_path = os.path.join(subfolder_paths[0], file)
    elif i < cut2:
        dest_path = os.path.join(subfolder_paths[1], file)
    else:
        dest_path = os.path.join(subfolder_paths[2], file)
    shutil.copy(src_path, dest_path)

# 为每个子文件夹重命名文件
for subfolder, subfolder_path in zip(subfolders, subfolder_paths):
    txt_file_path = os.path.join(folder_txt, f"{subfolder}.txt")

    # 读取TXT文件中的新名称
    with open(txt_file_path, 'r') as txt_file:
        new_names = txt_file.read().splitlines()

    # 获取子文件夹中的所有文件，并按名称排序
    subfolder_files = sorted([f for f in os.listdir(subfolder_path) if f.endswith('.mp4')],
                             key=lambda x: int(x.split('_')[1].split('.')[0]))

    # 检查新名称数量是否匹配文件数量
    if len(new_names) != len(subfolder_files):
        print(f"错误：文件夹 {subfolder_path} 中的文件数量与 {txt_file_path} 中的行数不匹配。")
        continue

    # 重命名文件
    for old_name, new_name in zip(subfolder_files, new_names):
        old_file_path = os.path.join(subfolder_path, old_name)
        new_file_path = os.path.join(subfolder_path, f"{new_name}.mp4")
        os.rename(old_file_path, new_file_path)

print("文件已成功分类、移动并重命名。")