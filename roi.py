import cv2
import numpy as np
import json
import os
import time

# --- 配置参数 ---
# 请替换为你的原始视频文件路径
VIDEO_PATH = (r"D:\24.1.3 CNO\25.1.3 CNO CeA #6"
              r""
              r" Cage2.mp4")

# 从视频中提取的帧号，用于作为ROI选择的背景图
FRAME_TO_EXTRACT = 100

# ROI配置文件的名称，用于保存和加载ROI定义
ROI_CONFIG_FILE = 'epm_rois.json'

# 图像显示尺寸参数 (用于 roi.py 内部的显示，方便用户点击)
# 如果原始视频宽度大于此值，ROI选择界面将缩放图像到此宽度进行显示。
# ！！重要提示：你在这里看到并点击的坐标，会根据这个显示宽度反向计算回原始视频分辨率！！
DISPLAY_WIDTH_FOR_GUI = 1000

# --- 全局变量 ---
points_display = []  # 当前正在选择的ROI在显示窗口上的点（显示坐标）
current_roi_name = ""
roi_definitions = {}  # 存储所有已定义的ROI (字典格式：{roi_name: [[x1_orig,y1_orig], ...]})
image_original_resolution = None  # 存储原始分辨率的背景图
image_display_resolution = None  # 存储用于显示和绘制的缩放后的图像副本
original_video_width = 0
original_video_height = 0
display_scale_factor = 1.0  # 用于将显示坐标反向缩放到原始坐标


# --- 视频帧提取函数 ---
def extract_frame_and_get_info(video_path, frame_number):
    """
    从视频中提取指定帧，并返回原始帧、宽度和高度。
    """
    if not os.path.exists(video_path):
        print(f"错误：视频文件 '{video_path}' 未找到。请检查路径是否正确。")
        return None, 0, 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 '{video_path}'。")
        return None, 0, 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if ret:
        height, width = frame.shape[:2]
        print(f"视频帧已成功提取。原始分辨率: {width}x{height}")
        return frame, width, height
    else:
        print(f"无法读取视频的第 {frame_number} 帧。请尝试调整 FRAME_TO_EXTRACT 或检查视频是否损坏。")
        return None, 0, 0


def mouse_callback(event, x, y, flags, param):
    """
    鼠标事件回调函数，用于捕获点击坐标。
    这些坐标是显示窗口上的坐标。
    """
    global points_display, image_display_resolution

    if event == cv2.EVENT_LBUTTONDOWN:
        points_display.append([x, y])
        # 在图像上绘制点和线，实时反馈
        cv2.circle(image_display_resolution, (x, y), 5, (0, 255, 0), -1)  # 绿色点
        if len(points_display) > 1:
            cv2.line(image_display_resolution, tuple(points_display[-2]), tuple(points_display[-1]), (0, 255, 0),
                     2)  # 绿色线
        cv2.imshow("Select ROI Points (Press 's' to save, 'c' to clear, 'q' to quit)", image_display_resolution)
        print(f"Point added: ({x}, {y}) [显示坐标]")


def start_roi_selection(background_frame_original, roi_name):
    """
    启动交互式ROI选择界面，并返回原始分辨率的ROI坐标。
    """
    global points_display, current_roi_name, image_display_resolution, roi_definitions
    global original_video_width, original_video_height, display_scale_factor

    current_roi_name = roi_name
    points_display = []  # 重置当前ROI的点列表

    img_to_display = background_frame_original.copy()

    # --- 图像缩放处理 (仅用于显示，不影响保存的坐标) ---
    original_video_height, original_video_width = img_to_display.shape[:2]

    if original_video_width > DISPLAY_WIDTH_FOR_GUI:
        new_width = DISPLAY_WIDTH_FOR_GUI
        new_height = int(original_video_height * (new_width / original_video_width))
        img_to_display = cv2.resize(img_to_display, (new_width, new_height), interpolation=cv2.INTER_AREA)
        display_scale_factor = new_width / original_video_width
        print(
            f"图像已从 ({original_video_width}x{original_video_height}) 缩放至 ({new_width}x{new_height}) 进行显示。缩放因子: {display_scale_factor:.4f}")
    else:
        display_scale_factor = 1.0  # 未缩放
        print(f"图像 (原始尺寸 {original_video_width}x{original_video_height}) 未缩放。")

    image_display_resolution = img_to_display.copy()  # 用于显示和绘制的图像副本

    cv2.namedWindow("Select ROI Points (Press 's' to save, 'c' to clear, 'q' to quit)")
    cv2.setMouseCallback("Select ROI Points (Press 's' to save, 'c' to clear, 'q' to quit)", mouse_callback)

    print(f"\n--- Selecting points for '{current_roi_name}' ---")
    print("左键点击以添加点。")
    print("按 's' 保存当前 ROI。")
    print("按 'c' 清除当前 ROI 的所有点并重新选择。")
    print("按 'q' 退出程序。")

    while True:
        temp_img = image_display_resolution.copy()  # 每次循环都从原始副本开始绘制，避免叠加

        # 绘制所有已保存的ROI (红色边框)
        # 注意：这里需要将保存的原始分辨率坐标转换回显示分辨率才能绘制
        for name, coords_orig_res in roi_definitions.items():
            if coords_orig_res:
                coords_display_res = [[int(p[0] * display_scale_factor), int(p[1] * display_scale_factor)] for p in
                                      coords_orig_res]
                pts = np.array(coords_display_res, np.int32).reshape((-1, 1, 2))
                cv2.polylines(temp_img, [pts], True, (0, 0, 255), 2)  # 红色边框
                cv2.putText(temp_img, name, (coords_display_res[0][0], coords_display_res[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 绘制当前正在选择的点的连线和点 (绿色点，蓝色预览线)
        if len(points_display) > 0:
            for i, p in enumerate(points_display):
                cv2.circle(temp_img, tuple(p), 5, (0, 255, 0), -1)  # 绿色点
                if i > 0:
                    cv2.line(temp_img, tuple(points_display[i - 1]), tuple(p), (0, 255, 0), 2)  # 绿色线
            if len(points_display) > 2:  # 如果有3个或更多点，绘制预览多边形
                pts_preview = np.array(points_display, np.int32).reshape((-1, 1, 2))
                cv2.polylines(temp_img, [pts_preview], True, (255, 0, 0), 1)  # 蓝色预览线

        cv2.imshow("Select ROI Points (Press 's' to save, 'c' to clear, 'q' to quit)", temp_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if len(points_display) >= 3:
                # --- 关键步骤：将显示坐标反向缩放回原始分辨率 ---
                points_original_resolution = []
                for p_display_x, p_display_y in points_display:
                    p_original_x = int(p_display_x / display_scale_factor)
                    p_original_y = int(p_display_y / display_scale_factor)
                    points_original_resolution.append([p_original_x, p_original_y])

                roi_definitions[current_roi_name] = points_original_resolution[:]
                print(f"ROI '{current_roi_name}' saved with {len(points_original_resolution)} points. (原始分辨率坐标)")
                break  # 退出当前ROI的选择循环
            else:
                print("需要至少3个点才能保存为一个多边形ROI。")
        elif key == ord('c'):
            points_display = []  # 清除当前ROI的点
            # 重新加载原始缩放图像以清除所有绘制，确保干净重选
            img_clean = background_frame_original.copy()
            if original_video_width > DISPLAY_WIDTH_FOR_GUI:
                new_width = DISPLAY_WIDTH_FOR_GUI
                new_height = int(original_video_height * (new_width / original_video_width))
                img_clean = cv2.resize(img_clean, (new_width, new_height), interpolation=cv2.INTER_AREA)
            image_display_resolution = img_clean.copy()
            print("Current points cleared. Please re-select.")
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return None  # 用户选择退出整个程序

    # 返回当前ROI的原始分辨率坐标
    return roi_definitions.get(current_roi_name)


def save_rois_to_json(rois, filename):
    """
    将ROI定义（原始分辨率坐标）保存到JSON文件。
    """
    with open(filename, 'w') as f:
        json.dump(rois, f, indent=4)
    print(f"ROI definitions saved to {filename}")


def load_rois_from_json(filename):
    """
    从JSON文件加载ROI定义（原始分辨率坐标）。
    """
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                rois = json.load(f)
            print(f"ROI definitions loaded from {filename}")
            return rois
        except json.JSONDecodeError:
            print(f"警告: ROI文件 '{filename}' 损坏或格式不正确，将创建一个新的。")
            return {}  # 返回空字典，重新定义
    return {}  # 文件不存在，返回空字典


if __name__ == "__main__":
    # --- 步骤1: 提取视频帧作为背景图 ---
    background_frame, original_video_width, original_video_height = extract_frame_and_get_info(VIDEO_PATH,
                                                                                               FRAME_TO_EXTRACT)
    if background_frame is None:
        print("无法继续，请确保视频路径正确且视频可读。")
        exit()

    # 尝试加载之前保存的ROI，以便在选择新的ROI时能看到旧的
    roi_definitions = load_rois_from_json(ROI_CONFIG_FILE)

    # --- 步骤2: 启动ROI选择GUI ---
    # 定义需要选择的ROI名称列表
    roi_names_to_select = ["center_zone", "open_arm1", "open_arm2"]  # 根据你的EPM迷宫，可能还需要 "closed_arm1", "closed_arm2" 等

    for roi_name in roi_names_to_select:
        # 传入原始分辨率的背景帧给选择函数
        coords = start_roi_selection(background_frame, roi_name)
        if coords is None:  # 用户按 'q' 退出
            print("用户退出 ROI 选择。")
            cv2.destroyAllWindows()
            exit()
        save_rois_to_json(roi_definitions, ROI_CONFIG_FILE)

    print("\n--- All ROIs Defined ---")
    print(json.dumps(roi_definitions, indent=4))

    cv2.destroyAllWindows()