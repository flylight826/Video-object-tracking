import cv2
import gc
import numpy as np
import os
import os.path as osp
import pdb
import torch
from sam2.build_sam import build_sam2_video_predictor
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def load_lasot_gt(gt_path):
    with open(gt_path, 'r') as f:
        gt = f.readlines()

    # bbox in first frame are prompts
    prompts = {}
    fid = 0
    for line in gt:
        line = line.strip()  # 移除首尾空白字符
        if not line:  # 跳过空行
            continue

        # 分割并处理浮点数
        parts = line.split(',')
        if len(parts) >= 4:
            try:
                # 先转换为浮点数，再转换为整数（四舍五入）
                x = float(parts[0])
                y = float(parts[1])
                w = float(parts[2])
                h = float(parts[3])

                # 四舍五入转换为整数
                x, y, w, h = map(round, [x, y, w, h])
                x, y, w, h = int(x), int(y), int(w), int(h)

                prompts[fid] = ((x, y, x + w, y + h), 0)
                fid += 1
            except ValueError as e:
                print(f"警告: 无法解析行 '{line}': {e}")
                continue

    print(f"成功加载 {len(prompts)} 个标注框")
    return prompts


color = [
    (255, 0, 0),
]

exp_name = "samurai"
model_name = "large"

checkpoint = f"sam2/checkpoints/sam2.1_hiera_{model_name}.pt"
if model_name == "base_plus":
    model_cfg = "configs/samurai/sam2.1_hiera_b+.yaml"
else:
    model_cfg = f"configs/samurai/sam2.1_hiera_{model_name[0]}.yaml"

video_folder = "/home/bls/new/samurai-master_new/data/lasot"  #set /path/to/data/lasot
pred_folder = f"results/{exp_name}/{exp_name}_{model_name}"

save_to_video = True
if save_to_video:
    vis_folder = f"visualization/{exp_name}/{model_name}"
    os.makedirs(vis_folder, exist_ok=True)

# 添加详细的调试信息
print(f"当前工作目录: {os.getcwd()}")
print(f"视频文件夹路径: {video_folder}")
print(f"视频文件夹是否存在: {osp.exists(video_folder)}")

if osp.exists(video_folder):
    print(f"视频文件夹内容: {os.listdir(video_folder)}")
else:
    print("错误: 视频文件夹不存在!")
    exit(1)

# 方法1: 尝试自动检测序列
all_sequences = []
if osp.exists(video_folder):
    # 获取所有子文件夹
    for item in os.listdir(video_folder):
        item_path = osp.join(video_folder, item)
        if osp.isdir(item_path):
            # 检查是否有color子文件夹和groundtruth.txt文件
            color_path = osp.join(item_path, "color")
            gt_path = osp.join(item_path, "groundtruth.txt")

            if osp.exists(color_path) and osp.exists(gt_path):
                all_sequences.append(item)
                print(f"找到序列: {item}")

# 方法2: 如果自动检测失败，尝试001-050
if len(all_sequences) == 0:
    print("自动检测失败，尝试001-050序列...")
    for i in range(1, 51):
        seq_name = f"{i:03d}"
        seq_path = osp.join(video_folder, seq_name)
        if osp.exists(seq_path):
            all_sequences.append(seq_name)
            print(f"找到序列: {seq_name}")

# 方法3: 如果仍然没有找到，尝试列出所有可能的文件夹
if len(all_sequences) == 0:
    print("尝试列出所有可能的文件夹...")
    for item in os.listdir(video_folder):
        item_path = osp.join(video_folder, item)
        if osp.isdir(item_path):
            print(f"找到文件夹: {item}")
            # 检查文件夹内容
            try:
                subitems = os.listdir(item_path)
                print(f"  文件夹内容: {subitems}")

                # 检查是否有color文件夹
                if "color" in subitems:
                    color_items = os.listdir(osp.join(item_path, "color"))
                    print(f"  color文件夹内容: {color_items[:5]}...")  # 只显示前5个

                # 检查是否有groundtruth.txt
                if "groundtruth.txt" in subitems:
                    print(f"  找到groundtruth.txt")

            except Exception as e:
                print(f"  读取文件夹内容出错: {e}")

print(f"最终找到 {len(all_sequences)} 个序列: {sorted(all_sequences)}")

if len(all_sequences) == 0:
    print("错误: 没有找到任何序列! 请检查数据路径和结构。")
    exit(1)

# 创建预测结果文件夹
os.makedirs(pred_folder, exist_ok=True)

for vid, video in enumerate(all_sequences):
    # 直接使用序列名称
    video_basename = video

    # 为每个序列创建子文件夹（仅预测结果）
    seq_pred_folder = osp.join(pred_folder, video_basename)
    os.makedirs(seq_pred_folder, exist_ok=True)

    # 修改frame_folder路径，指向color文件夹
    frame_folder = osp.join(video_folder, video, "color")

    # 添加序列路径调试信息
    print(f"\n处理序列 {video}:")
    print(f"  完整路径: {frame_folder}")
    print(f"  路径是否存在: {osp.exists(frame_folder)}")

    if osp.exists(frame_folder):
        try:
            files = os.listdir(frame_folder)
            print(f"  color文件夹内容: {len(files)} 个文件")
            png_files = [f for f in files if f.endswith('.png')]
            print(f"  PNG文件数量: {len(png_files)}")

            if len(png_files) > 0:
                print(f"  前5个PNG文件: {sorted(png_files)[:5]}")
        except Exception as e:
            print(f"  读取color文件夹出错: {e}")
            continue
    else:
        print(f"  错误: 序列 {video} 的color文件夹不存在!")
        continue

    # 检查序列是否存在
    if not osp.exists(frame_folder):
        print(f"Warning: Sequence {video} not found, skipping...")
        continue

    # 获取PNG图片数量
    png_files = [f for f in os.listdir(frame_folder) if f.endswith('.png')]
    num_frames = len(png_files)

    if num_frames == 0:
        print(f"Warning: No PNG files found in {frame_folder}, skipping...")
        continue

    print(f"\033[91mRunning video [{vid + 1}/{len(all_sequences)}]: {video} with {num_frames} frames\033[0m")

    # 获取第一张图片的尺寸
    try:
        first_frame_path = osp.join(frame_folder, sorted(png_files)[0])
        img = cv2.imread(first_frame_path)
        if img is None:
            print(f"错误: 无法读取第一帧图片 {first_frame_path}")
            continue
        height, width = img.shape[:2]
        print(f"  图片尺寸: {width}x{height}")
    except Exception as e:
        print(f"  读取第一帧图片出错: {e}")
        continue

    # 检查预测文件是否已存在（避免重复处理）
    output_file = osp.join(seq_pred_folder, f'{video_basename}.txt')
    if osp.exists(output_file):
        print(f"  预测文件已存在，跳过序列 {video}")
        continue

    try:
        predictor = build_sam2_video_predictor(model_cfg, checkpoint, device="cuda:0")
    except Exception as e:
        print(f"  初始化predictor出错: {e}")
        continue

    predictions = []

    if save_to_video:
        # 可视化视频保持原样，不创建子文件夹
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(osp.join(vis_folder, f'{video_basename}.mp4'), fourcc, 30, (width, height))

    # Start processing frames
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            # 修复：移除不支持的 image_extension 参数
            state = predictor.init_state(frame_folder, offload_video_to_cpu=True,
                                         offload_state_to_cpu=True, async_loading_frames=True)

            # 修改标注文件路径
            gt_path = osp.join(video_folder, video, "groundtruth.txt")
            print(f"  标注文件路径: {gt_path}")
            print(f"  标注文件是否存在: {osp.exists(gt_path)}")

            if not osp.exists(gt_path):
                print(f"  错误: 标注文件不存在!")
                continue

            # 在读取标注文件前先检查内容
            try:
                with open(gt_path, 'r') as f:
                    first_line = f.readline().strip()
                    print(f"  标注文件第一行内容: '{first_line}'")
            except Exception as e:
                print(f"  读取标注文件第一行出错: {e}")

            prompts = load_lasot_gt(gt_path)
            print(f"  加载了 {len(prompts)} 个标注")

            if len(prompts) == 0:
                print(f"  错误: 没有成功加载任何标注!")
                continue

            bbox, track_label = prompts[0]
            frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=0)

            for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
                mask_to_vis = {}
                bbox_to_vis = {}

                assert len(masks) == 1 and len(object_ids) == 1, "Only one object is supported right now"
                for obj_id, mask in zip(object_ids, masks):
                    mask = mask[0].cpu().numpy()
                    mask = mask > 0.0
                    non_zero_indices = np.argwhere(mask)
                    if len(non_zero_indices) == 0:
                        bbox = [0, 0, 0, 0]
                    else:
                        y_min, x_min = non_zero_indices.min(axis=0).tolist()
                        y_max, x_max = non_zero_indices.max(axis=0).tolist()
                        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                    bbox_to_vis[obj_id] = bbox
                    mask_to_vis[obj_id] = mask

                if save_to_video:
                    # 修改为使用PNG图片
                    frame_filename = f"{frame_idx + 1:08d}.png"
                    img_path = osp.join(frame_folder, frame_filename)
                    img = cv2.imread(img_path)

                    if img is None:
                        print(f"Warning: Could not load frame {img_path}")
                        break

                    for obj_id in mask_to_vis.keys():
                        mask_img = np.zeros((height, width, 3), np.uint8)
                        mask_img[mask_to_vis[obj_id]] = color[(obj_id + 1) % len(color)]
                        img = cv2.addWeighted(img, 1, mask_img, 0.75, 0)

                    for obj_id in bbox_to_vis.keys():
                        cv2.rectangle(img, (bbox_to_vis[obj_id][0], bbox_to_vis[obj_id][1]),
                                      (bbox_to_vis[obj_id][0] + bbox_to_vis[obj_id][2],
                                       bbox_to_vis[obj_id][1] + bbox_to_vis[obj_id][3]),
                                      color[(obj_id) % len(color)], 2)

                    # 检查当前帧是否有标注
                    if frame_idx in prompts:
                        x1, y1, x2, y2 = prompts[frame_idx][0]
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    out.write(img)

                predictions.append(bbox_to_vis)

        # 保存预测结果，改为整数格式
        with open(output_file, 'w') as f:
            for pred in predictions:
                if 0 in pred:  # 检查对象ID 0是否存在
                    x, y, w, h = pred[0]
                    # 直接保存为整数
                    f.write(f"{int(x)},{int(y)},{int(w)},{int(h)}\n")
                else:
                    # 如果没有检测到对象，写入全零
                    f.write("0,0,0,0\n")
        print(f"  预测结果已保存到: {output_file}")

        if save_to_video:
            out.release()
            print(f"  视频已保存到: {osp.join(vis_folder, f'{video_basename}.mp4')}")

    except Exception as e:
        print(f"  处理序列 {video} 时出错: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # 确保资源被释放
        if 'predictor' in locals():
            del predictor
        if 'state' in locals():
            del state
        if save_to_video and 'out' in locals():
            out.release()
        gc.collect()
        torch.clear_autocast_cache()
        torch.cuda.empty_cache()

print("所有序列处理完成!")

