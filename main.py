import pandas as pd
from shapely.geometry import Polygon, MultiPolygon, Point
import matplotlib.pyplot as plt
import json
import os
import numpy as np # 确保导入了numpy，虽然可能已经有了

# --- 数据准备与理解部分 ---
# (这部分应该与你之前main.py中的代码相同，确保数据加载和列清理正确)

# 加载数据
try:
    df = pd.read_csv('25.1.3 CNO CeA #6 Cage2DLC_resnet50_EPMJul23shuffle1_15000_filtered.csv', header=[1, 2])
    print("数据加载成功！")
except FileNotFoundError:
    print("错误：数据文件未找到。请确保CSV文件与脚本在同一目录下，或者提供正确的文件路径。")
    exit()

# 清理列名，使其更易于访问
bodyparts = df.columns.get_level_values(0).unique().tolist()
if 'scorer' in bodyparts:
    bodyparts.remove('scorer')

# 创建一个新的DataFrame，简化列名
df.columns = ['_'.join(col).strip() for col in df.columns.values]
df = df.iloc[2:] # 移除前两行，它们是scorer和bodyparts的描述
df = df.apply(pd.to_numeric, errors='coerce') # 将数据转换为数值类型
df.reset_index(drop=True, inplace=True) # 重置索引

# --- 计算小鼠身体重心 (使用 nose 和 neck_base 的简单平均) ---
selected_body_parts_for_average = ['nose', 'neck_base']

# 检查所有必需的列是否存在
required_x_cols = [f'{part}_x' for part in selected_body_parts_for_average]
required_y_cols = [f'{part}_y' for part in selected_body_parts_for_average]
missing_cols = [col for col in required_x_cols + required_y_cols if col not in df.columns]

if missing_cols:
    print(f"警告：DataFrame中缺少用于计算重心的以下列：{missing_cols}。")
    if 'neck_base_x' in df.columns and 'neck_base_y' in df.columns:
        print("将回退到仅使用 'neck_base' 作为小鼠位置。")
        mouse_x = df['neck_base_x']
        mouse_y = df['neck_base_y']
        likelihood = df['neck_base_likelihood']
        print(f"\n小鼠位置数据 (回退到仅使用 neck_base):")
    else:
        print("错误：没有足够的身体点（nose 或 neck_base）来计算小鼠位置。请检查数据。")
        exit()
else:
    points_to_average_x = df[required_x_cols]
    points_to_average_y = df[required_y_cols]

    mouse_x = points_to_average_x.mean(axis=1)
    mouse_y = points_to_average_y.mean(axis=1)
    likelihood = df['neck_base_likelihood'] # 仍然使用 neck_base 的置信度作为代表

    print(f"\n小鼠位置数据 (身体重心 - {selected_body_parts_for_average} 的简单平均):")

print(f"X坐标前5行:\n{mouse_x.head()}")
print(f"Y坐标前5行:\n{mouse_y.head()}")
print(f"置信度前5行:\n{likelihood.head()}")

# --- ROI 加载与定义逻辑 ---
# 定义ROI配置文件路径
ROI_CONFIG_FILE = 'epm_rois.json'

def load_rois_from_json(filename):
    """
    从JSON文件加载ROI定义。
    假设 roi.py 已经将 ROI 坐标保存为原始视频分辨率的坐标。
    """
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                rois = json.load(f)
            print(f"ROI definitions loaded from {filename}")
            return rois
        except json.JSONDecodeError:
            print(f"错误：ROI配置文件 '{filename}' 损坏或格式不正确。请重新运行 'roi.py' 定义ROI。")
            raise FileNotFoundError(f"ROI file '{filename}' corrupted. Please run roi.py first.")
    else:
        print(f"错误：ROI配置文件 '{filename}' 不存在。请先单独运行 'roi.py' 脚本来定义你的 ROIs。")
        raise FileNotFoundError(f"ROI file '{filename}' not found. Please run roi.py first.")

# 加载ROI，如果文件不存在或损坏，程序将在此处退出
try:
    loaded_rois = load_rois_from_json(ROI_CONFIG_FILE)
except FileNotFoundError:
    exit()

# 从加载的字典中创建 Shapely Polygon 对象
# 这些ROI坐标现在应该已经是原始视频分辨率的坐标了
center_zone_polygon = None
if "center_zone" in loaded_rois and loaded_rois["center_zone"]:
    center_zone_polygon = Polygon(loaded_rois["center_zone"])
else:
    print("警告：'center_zone' ROI 未在JSON中找到或为空。")

open_arm1_polygon = None
if "open_arm1" in loaded_rois and loaded_rois["open_arm1"]:
    open_arm1_polygon = Polygon(loaded_rois["open_arm1"])
else:
    print("警告：'open_arm1' ROI 未在JSON中找到或为空。")

open_arm2_polygon = None
if "open_arm2" in loaded_rois and loaded_rois["open_arm2"]:
    open_arm2_polygon = Polygon(loaded_rois["open_arm2"])
else:
    print("警告：'open_arm2' ROI 未在JSON中找到或为空。")

# 合并开放臂为一个 MultiPolygon
open_arms_polygons = []
if open_arm1_polygon:
    open_arms_polygons.append(open_arm1_polygon)
if open_arm2_polygon:
    open_arms_polygons.append(open_arm2_polygon)

open_arm_combined_polygon = None
if open_arms_polygons:
    if len(open_arms_polygons) > 1:
        open_arm_combined_polygon = MultiPolygon(open_arms_polygons)
    else:
        open_arm_combined_polygon = open_arms_polygons[0]
else:
    print("警告: 没有定义开放臂ROI。")


print("\nROI Polygon 对象创建完成：")
if center_zone_polygon:
    print(f"中心区多边形: {list(center_zone_polygon.exterior.coords)}")
if open_arm1_polygon:
    print(f"开放臂1多边形: {list(open_arm1_polygon.exterior.coords)}")
if open_arm2_polygon:
    print(f"开放臂2多边形: {list(open_arm2_polygon.exterior.coords)}")


# --- 可视化ROI并加入小鼠运动轨迹 ---
plt.figure(figsize=(12, 10))

# 绘制ROI区域的边框
if center_zone_polygon:
    plt.plot(*center_zone_polygon.exterior.xy, color='blue', linewidth=2, label='Center Zone')
if open_arm1_polygon:
    plt.plot(*open_arm1_polygon.exterior.xy, color='red', linewidth=2, label='Open Arm 1')
if open_arm2_polygon:
    plt.plot(*open_arm2_polygon.exterior.xy, color='red', linewidth=2, label='Open Arm 2')

# --- 加入小鼠完整运动轨迹的绘制 (直接使用原始 mouse_x, mouse_y 坐标) ---
# 注意：这里不再进行任何缩放
valid_trajectory_mask = mouse_x.notna() & mouse_y.notna()
plt.plot(mouse_x[valid_trajectory_mask], mouse_y[valid_trajectory_mask],
         color='gray', linewidth=0.5, alpha=0.7, label='Mouse Trajectory')

plt.xlabel('X Coordinate (Original Video Resolution)')
plt.ylabel('Y Coordinate (Original Video Resolution)')
plt.title('Elevated Plus Maze ROIs and Mouse Trajectory (Original Resolution)')
plt.legend()
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')

# 调整X和Y轴的显示范围，现在应该基于原始视频分辨率的坐标来计算
# 理想情况下，这应该匹配原始视频的宽高。
# 你可以手动设置，或者从视频文件中读取（如果需要更动态的）。
# 暂时用一个合理的默认值或从轨迹/ROI中找到最大最小值。
# 为了确保轨迹和ROI都在视图中，我们取它们所有坐标的最大最小值
all_x_coords = np.array([])
all_y_coords = np.array([])

if len(mouse_x[valid_trajectory_mask]) > 0:
    all_x_coords = np.append(all_x_coords, mouse_x[valid_trajectory_mask].dropna().values)
    all_y_coords = np.append(all_y_coords, mouse_y[valid_trajectory_mask].dropna().values)

if center_zone_polygon:
    all_x_coords = np.append(all_x_coords, center_zone_polygon.exterior.xy[0].tolist())
    all_y_coords = np.append(all_y_coords, center_zone_polygon.exterior.xy[1].tolist())
if open_arm1_polygon:
    all_x_coords = np.append(all_x_coords, open_arm1_polygon.exterior.xy[0].tolist())
    all_y_coords = np.append(all_y_coords, open_arm1_polygon.exterior.xy[1].tolist())
if open_arm2_polygon:
    all_x_coords = np.append(all_x_coords, open_arm2_polygon.exterior.xy[0].tolist())
    all_y_coords = np.append(all_y_coords, open_arm2_polygon.exterior.xy[1].tolist())

all_x_coords = all_x_coords[np.isfinite(all_x_coords)]
all_y_coords = all_y_coords[np.isfinite(all_y_coords)]


if len(all_x_coords) > 0 and len(all_y_coords) > 0:
    # 增加一些边距，确保所有内容都可见
    x_min, x_max = all_x_coords.min() - 50, all_x_coords.max() + 50
    y_min, y_max = all_y_coords.min() - 50, all_y_coords.max() + 50
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
else:
    print("警告：无法自动设置绘图边界，因为没有有效的轨迹或ROI坐标。请检查数据。")

plt.show()

# --- 判断小鼠位置与ROI关系 (直接使用原始 mouse_x 和 mouse_y 坐标) ---
def is_in_roi(x, y, polygon):
    if pd.isna(x) or pd.isna(y) or polygon is None:
        return False
    point = Point(x, y)
    return polygon.contains(point)

# 应用函数到DataFrame，创建新的列标记小鼠是否在各个区域内
# 这里直接使用原始的 mouse_x 和 mouse_y
df['in_center_zone'] = df.apply(lambda row: is_in_roi(mouse_x.loc[row.name], mouse_y.loc[row.name], center_zone_polygon), axis=1)
df['in_open_arm'] = df.apply(lambda row: is_in_roi(mouse_x.loc[row.name], mouse_y.loc[row.name], open_arm_combined_polygon), axis=1)


print("\n小鼠在ROI中的标记 (前5行):")
# 打印原始的重心坐标和判断结果，方便核对
print(df[['in_center_zone', 'in_open_arm']].head())
for i in range(min(5, len(df))):
    print(f"帧 {i}: 重心({mouse_x.iloc[i]:.2f}, {mouse_y.iloc[i]:.2f}), "
          f"在中心区: {df['in_center_zone'].iloc[i]}, 在开放臂: {df['in_open_arm'].iloc[i]}")

# --- 记录进入中心区和开放臂的时间 ---
fps = 30 # 假设视频帧率 (Frames Per Second)
frame_interval_sec = 1 / fps
# ... (main.py 文件中，直到计算 fps 和 frame_interval_sec 的部分保持不变) ...



# --- 新增部分：从用户读取分析时间范围 ---
analysis_start_sec = 0
analysis_end_sec = float('inf') # 默认到视频结束

while True:
    start_input = input("\n请输入分析的起始时间 (秒)，留空表示从0秒开始: ")
    if start_input == '':
        analysis_start_sec = 0
        break
    try:
        analysis_start_sec = float(start_input)
        if analysis_start_sec < 0:
            print("起始时间不能为负数，请重新输入。")
        else:
            break
    except ValueError:
        print("输入无效，请输入一个数字。")

while True:
    end_input = input("请输入分析的结束时间 (秒)，留空表示到视频结束: ")
    if end_input == '':
        # 如果留空，则计算总帧数对应的总时间作为结束时间
        # 确保 df 此时已经加载
        analysis_end_sec = len(df) * frame_interval_sec
        break
    try:
        analysis_end_sec = float(end_input)
        if analysis_end_sec <= analysis_start_sec:
            print(f"结束时间必须大于起始时间 ({analysis_start_sec:.2f}秒)，请重新输入。")
        else:
            break
    except ValueError:
        print("输入无效，请输入一个数字。")

print(f"\n将分析从 {analysis_start_sec:.2f} 秒到 {analysis_end_sec:.2f} 秒的数据。")

# 将时间范围转换为帧索引
start_frame_index = int(analysis_start_sec * fps)
end_frame_index = int(analysis_end_sec * fps)

# 确保索引在DataFrame的有效范围内
start_frame_index = max(0, start_frame_index)
end_frame_index = min(len(df), end_frame_index)

# --- 修改部分：对 DataFrame 进行时间范围切片 ---
# 创建一个在指定时间范围内有效的布尔掩码
# 注意：DataFrame的索引是0开始的帧号
time_range_mask = (df.index >= start_frame_index) & (df.index < end_frame_index)

# 使用这个掩码来过滤数据，只分析指定时间段内的行为
df_filtered = df[time_range_mask].copy()


# 重新计算在每个区域的总帧数，但现在基于过滤后的数据
total_frames_center_zone = df_filtered['in_center_zone'].sum()
total_frames_open_arm = df_filtered['in_open_arm'].sum()

# 计算总停留时间（秒）
time_in_center_zone_sec = total_frames_center_zone * frame_interval_sec
time_in_open_arm_sec = total_frames_open_arm * frame_interval_sec

print(f"\n分析结果 ({analysis_start_sec:.2f}s - {analysis_end_sec:.2f}s):")
print(f"在中心区停留的总时间: {time_in_center_zone_sec:.2f} 秒")
print(f"在开放臂停留的总时间: {time_in_open_arm_sec:.2f} 秒")

# 记录每次进入的时间（首次进入或从其他区域进入）
# 同样，基于过滤后的数据进行迭代，并调整时间戳
center_zone_entries = []
in_center_prev = False
# 遍历过滤后的DataFrame的索引，并计算实际时间
for i, in_center_curr in df_filtered['in_center_zone'].items():
    current_frame_actual_index = i # 这是原始DataFrame中的帧号
    if in_center_curr and not in_center_prev:
        center_zone_entries.append(current_frame_actual_index * frame_interval_sec)
    in_center_prev = in_center_curr

print(f"\n每次进入中心区的时间点 (秒) ({analysis_start_sec:.2f}s - {analysis_end_sec:.2f}s):")
print([f"{t:.2f}" for t in center_zone_entries[:10]] + (['...'] if len(center_zone_entries) > 10 else []))

open_arm_entries = []
in_open_prev = False
for i, in_open_curr in df_filtered['in_open_arm'].items():
    current_frame_actual_index = i # 这是原始DataFrame中的帧号
    if in_open_curr and not in_open_prev:
        open_arm_entries.append(current_frame_actual_index * frame_interval_sec)
    in_open_prev = in_open_curr

print(f"\n每次进入开放臂的时间点 (秒) ({analysis_start_sec:.2f}s - {analysis_end_sec:.2f}s):")
print([f"{t:.2f}" for t in open_arm_entries[:10]] + (['...'] if len(open_arm_entries) > 10 else []))
# 计算在每个区域的总帧数
total_frames_center_zone = df['in_center_zone'].sum()
total_frames_open_arm = df['in_open_arm'].sum()

# 计算总停留时间（秒）
time_in_center_zone_sec = total_frames_center_zone * frame_interval_sec
time_in_open_arm_sec = total_frames_open_arm * frame_interval_sec

print(f"\n分析结果:")
print(f"在中心区停留的总时间: {time_in_center_zone_sec:.2f} 秒")
print(f"在开放臂停留的总时间: {time_in_open_arm_sec:.2f} 秒")

# 记录每次进入的时间（首次进入或从其他区域进入）
center_zone_entries = []
in_center_prev = False
for i, in_center_curr in enumerate(df['in_center_zone']):
    if in_center_curr and not in_center_prev:
        center_zone_entries.append(i * frame_interval_sec)
    in_center_prev = in_center_curr

print(f"\n每次进入中心区的时间点 (秒):")
print([f"{t:.2f}" for t in center_zone_entries[:10]] + (['...'] if len(center_zone_entries) > 10 else []))

open_arm_entries = []
in_open_prev = False
for i, in_open_curr in enumerate(df['in_open_arm']):
    if in_open_curr and not in_open_prev:
        open_arm_entries.append(i * frame_interval_sec)
    in_open_prev = in_open_curr

print(f"\n每次进入开放臂的时间点 (秒):")
print([f"{t:.2f}" for t in open_arm_entries[:10]] + (['...'] if len(open_arm_entries) > 10 else []))

# ... (前面的代码保持不变，包括用户输入分析时间范围的逻辑) ...

print(f"\n分析结果 ({analysis_start_sec:.2f}s - {analysis_end_sec:.2f}s):")
print(f"在中心区停留的总时间: {time_in_center_zone_sec:.2f} 秒")
print(f"在开放臂停留的总时间: {time_in_open_arm_sec:.2f} 秒")

