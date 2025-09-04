import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import cv2  # Used for reading video frames
import os  # Used for path operations
import matplotlib  # Used for font configuration

matplotlib.use('TkAgg')  # Or 'QtAgg', 'WXAgg' depending on your backend preference

# --- Configure matplotlib for Chinese display ---
# Replace with the path or name of a Chinese TrueType font on your system
# e.g., 'SimHei', 'Microsoft YaHei', 'PingFang SC', 'WenQuanYi Zen Hei'
# You can also specify the full font path, e.g., 'C:/Windows/Fonts/simhei.ttf'
font_path = 'SimHei'  # Attempt to use a common black body in the system

try:
    matplotlib.rcParams['font.sans-serif'] = [font_path,
                                              'DejaVu Sans']  # Prioritize Chinese font, fallback to DejaVu Sans
    matplotlib.rcParams['axes.unicode_minus'] = False  # Resolve the issue of negative signs displaying as squares
    print(f"Matplotlib font set to: {matplotlib.rcParams['font.sans-serif']}")
except Exception as e:
    print(f"Warning: Failed to set Matplotlib font, possibly because font '{font_path}' does not exist. Error: {e}")
    print("Please manually check if the font exists on your system, or try other font names/paths.")
# --- End font configuration ---


# --- Configuration Parameters ---
fps = 50  # Frames per second
likelihood_threshold = 0.60  # Likelihood threshold, points below this value will be filtered out

# Video file associated with the CSV (needed for ROI selection)
video_file = r"D:\CeA Crh Microbio\Day11 OFT\DJI_20250708171042_0021_D.MP4" # Example video path -
# IMPORTANT: You need to set this path!
file_to_analyze = r"D:\CeA Crh Microbio\Day -1 OFT\19DLC_resnet50_third trialJul21shuffle1_12000_el.csv"  # Example CSV path

# --- Body Part Column Name Mapping ---
# *** IMPORTANT: Ensure these map correctly to your DeepLabCut CSV file's second header row. ***
# If your CSV uses 'nose', 'bodycenter', 'tailbase' directly, then map them as such.
# If it uses 'bodypart1', 'bodypart2', 'bodypart3', then map them accordingly.
BODYPART_COLUMN_MAP = {
    'nose': 'bodypart1',  # Assuming 'bodypart1' in CSV is nose
    'bodycenter': 'bodypart2',  # Assuming 'bodypart2' in CSV is bodycenter
    'tailbase': 'bodypart3',  # Assuming 'bodypart3' in CSV is tailbase
}


# --- Helper Function: Get individual's frame-by-frame status in a zone ---
def get_individual_zone_status(df_data, individual_prefix, zone_details, bodyparts_to_consider_for_centroid,
                               likelihood_threshold):
    """
    Calculates the frame-by-frame centroid coordinates and in-zone boolean status for a single animal
    using a prioritized fallback logic for centroid calculation.

    Args:
        df_data (pd.DataFrame): DataFrame containing DeepLabCut tracking data (already filtered by analysis time).
        individual_prefix (str): Prefix for the animal (e.g., 'individual1').
        zone_details (dict): Dictionary containing zone 'x_min', 'x_max', 'y_min', 'y_max'.
        bodyparts_to_consider_for_centroid (list): List of body part names to potentially use for centroid calculation.
                                                    Should be ['nose', 'bodycenter', 'tailbase'] for this logic.
        likelihood_threshold (float): Likelihood threshold.

    Returns:
        tuple: (pd.Series, pd.Series, pd.Series, list) containing centroid x-coordinates, y-coordinates,
               in-zone boolean Series, and list of original frame indices.
               Returns empty Series and list if no valid data.
    """
    centroid_x_coords = []
    centroid_y_coords = []
    centroid_likelihoods = []

    # Expected body parts for the specific fallback logic
    NOSE_KEY = 'nose'
    BODYCENTER_KEY = 'bodycenter'
    TAILBASE_KEY = 'tailbase'

    for index, _ in df_data.iterrows():
        # Store valid body part coordinates for the current frame
        valid_bp_data = {}  # {bp_name: {'x': x_val, 'y': y_val, 'likelihood': l_val}}

        # 1. Collect valid data for all relevant body parts in the current frame
        for bp_name in bodyparts_to_consider_for_centroid:
            bp_col_prefix = BODYPART_COLUMN_MAP.get(bp_name)
            if bp_col_prefix is None:
                continue

            try:
                x_val = pd.to_numeric(df_data.loc[index, (individual_prefix, bp_col_prefix)], errors='coerce')
                y_val = pd.to_numeric(df_data.loc[index, (individual_prefix, f'{bp_col_prefix}.1')], errors='coerce')
                likelihood_val = pd.to_numeric(df_data.loc[index, (individual_prefix, f'{bp_col_prefix}.2')],
                                               errors='coerce')

                if pd.notna(x_val) and pd.notna(y_val) and pd.notna(
                        likelihood_val) and likelihood_val >= likelihood_threshold:
                    valid_bp_data[bp_name] = {'x': x_val, 'y': y_val, 'likelihood': likelihood_val}
            except KeyError:
                pass  # Column not found for this individual/body part
            except Exception as e:
                print(f"Warning: Unknown error reading {individual_prefix}'s {bp_name} data at frame {index}: {e}")

        # 2. Apply fallback logic to calculate centroid for the current frame
        current_centroid_x, current_centroid_y, current_centroid_likelihood = np.nan, np.nan, np.nan

        # Priority 1: Nose AND Bodycenter exist
        if NOSE_KEY in valid_bp_data and BODYCENTER_KEY in valid_bp_data:
            x_vals = [valid_bp_data[NOSE_KEY]['x'], valid_bp_data[BODYCENTER_KEY]['x']]
            y_vals = [valid_bp_data[NOSE_KEY]['y'], valid_bp_data[BODYCENTER_KEY]['y']]
            l_vals = [valid_bp_data[NOSE_KEY]['likelihood'], valid_bp_data[BODYCENTER_KEY]['likelihood']]
            current_centroid_x = np.mean(x_vals)
            current_centroid_y = np.mean(y_vals)
            current_centroid_likelihood = np.mean(l_vals)
        # Priority 2: Only Nose exists
        elif NOSE_KEY in valid_bp_data:
            current_centroid_x = valid_bp_data[NOSE_KEY]['x']
            current_centroid_y = valid_bp_data[NOSE_KEY]['y']
            current_centroid_likelihood = valid_bp_data[NOSE_KEY]['likelihood']
        # Priority 3: Bodycenter AND Tailbase exist (and nose does not)
        elif BODYCENTER_KEY in valid_bp_data and TAILBASE_KEY in valid_bp_data:
            x_vals = [valid_bp_data[BODYCENTER_KEY]['x'], valid_bp_data[TAILBASE_KEY]['x']]
            y_vals = [valid_bp_data[BODYCENTER_KEY]['y'], valid_bp_data[TAILBASE_KEY]['y']]
            l_vals = [valid_bp_data[BODYCENTER_KEY]['likelihood'], valid_bp_data[TAILBASE_KEY]['likelihood']]
            current_centroid_x = np.mean(x_vals)
            current_centroid_y = np.mean(y_vals)
            current_centroid_likelihood = np.mean(l_vals)
        # Fallback: No sufficient valid points
        else:
            # Centroid remains NaN
            pass

        centroid_x_coords.append(current_centroid_x)
        centroid_y_coords.append(current_centroid_y)
        centroid_likelihoods.append(current_centroid_likelihood)

    x_coords = pd.Series(centroid_x_coords, index=df_data.index)
    y_coords = pd.Series(centroid_y_coords, index=df_data.index)
    likelihoods = pd.Series(centroid_likelihoods, index=df_data.index)

    valid_data_mask = x_coords.notna() & y_coords.notna() & likelihoods.notna()

    x_coords_cleaned = x_coords[valid_data_mask]
    y_coords_cleaned = y_coords[valid_data_mask]
    likelihoods_cleaned = likelihoods[valid_data_mask]

    cleaned_rows = len(x_coords_cleaned)
    original_frame_indices = x_coords_cleaned.index.to_list()

    if cleaned_rows == 0:
        return pd.Series(dtype='float64'), pd.Series(dtype='float64'), pd.Series(dtype='bool'), []

    in_zone = (x_coords_cleaned >= zone_details["x_min"]) & \
              (x_coords_cleaned <= zone_details["x_max"]) & \
              (y_coords_cleaned >= zone_details["y_min"]) & \
              (y_coords_cleaned <= zone_details["y_max"])

    return x_coords_cleaned, y_coords_cleaned, in_zone, original_frame_indices


# --- Main function: Calculate individual dwelling time in a zone (now receives in_zone mask) ---
def calculate_zone_dwelling_time(in_zone_mask, original_frame_indices, fps, individual_prefix, zone_display_name):
    """
    Calculates the dwelling time for an individual in a specific zone based on the given in_zone mask
    and original frame indices.

    Args:
        in_zone_mask (pd.Series): Boolean Series indicating whether the individual is in the zone for each frame.
        original_frame_indices (list): List of original frame indices corresponding to in_zone_mask.
        fps (int): Frames per second.
        individual_prefix (str): Prefix for the animal (e.g., 'individual1').
        zone_display_name (str): Display name for the zone (for print output).

    Returns:
        tuple: (pd.DataFrame, float) containing a DataFrame of entry and dwelling times, and total dwelling time in seconds.
    """
    print(f"\n--- Analyzing {individual_prefix} in {zone_display_name} ---")

    entry_times_frames = []
    entry_times_seconds = []
    duration_frames = []
    duration_seconds = []

    is_in_zone = False
    current_entry_frame = 0

    # Ensure original_frame_indices is not empty
    if not original_frame_indices:
        print(f"No valid frames for {individual_prefix} in {zone_display_name}.")
        return pd.DataFrame(), 0.0

    for i, status in enumerate(in_zone_mask):
        # This check should ideally not be needed if in_zone_mask and original_frame_indices are always aligned
        if i >= len(original_frame_indices):
            print(f"original_frame_indices out of bounds at i={i}. Skipping.")
            continue

        current_original_frame = original_frame_indices[i]

        if status and not is_in_zone:
            is_in_zone = True
            current_entry_frame = current_original_frame
            entry_times_frames.append(current_entry_frame)
            entry_times_seconds.append(current_entry_frame / fps)
        elif not status and is_in_zone:
            is_in_zone = False
            duration_frames.append(current_original_frame - current_entry_frame)
            duration_seconds.append((current_original_frame - current_entry_frame) / fps)

    if is_in_zone:  # If still in zone at the end of data
        last_frame_original_index = original_frame_indices[-1]
        duration_frames.append(last_frame_original_index - current_entry_frame + 1)
        duration_seconds.append((last_frame_original_index - current_entry_frame + 1) / fps)

    results_df = pd.DataFrame({
        '进入帧': entry_times_frames,
        '进入时间 (s)': entry_times_seconds,
        '持续帧数': duration_frames,
        '持续时间 (s)': duration_seconds
    })

    total_time_in_zone_seconds = results_df['持续时间 (s)'].sum()
    print(results_df.to_string(index=False))
    print(f"总计在 {zone_display_name} 停留时间： {total_time_in_zone_seconds:.2f} 秒")

    return results_df, total_time_in_zone_seconds


# --- GUI function for ROI selection (unique and complete version) ---
def select_roi(video_path, frame_to_capture=1000):
    """
    Captures an image from a specific frame of the specified video and allows the user to select a
    rectangular ROI by clicking two diagonal points. Returns the coordinates of the selected rectangle
    (x_min, y_min, x_max, y_max).
    """
    print(f"Attempting to open video file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}. Please check the path and if the file exists.")
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_capture)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(
            f"Error: Unable to read frame {frame_to_capture} of the video. The frame number might be out of total video frames or video opening failed.")
        return None

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots()
    ax.imshow(frame_rgb)
    ax.set_title(f"On video frame {frame_to_capture}: Please click two diagonal points to define the open field.")
    plt.axis('off')

    clicked_points = []
    final_roi_coords = None
    temp_rect_patch = None  # For drawing temporary rectangle

    def onclick(event):
        nonlocal clicked_points, final_roi_coords, temp_rect_patch
        print(
            f"Mouse click event occurred: button={event.button}, xdata={event.xdata}, ydata={event.ydata}, inaxes={event.inaxes}")

        if event.inaxes != ax:
            print("Click not within image area, ignoring.")
            return

        if event.button == 1:  # Left mouse button click
            x, y = event.xdata, event.ydata
            if x is None or y is None:
                print("Invalid click position (None).")
                return

            clicked_points.append((x, y))
            print(f"Captured click point {len(clicked_points)}: ({x:.0f}, {y:.0f})")

            # Clear old drawn points, ensure only current click or final rectangle is shown
            for artist in ax.lines:
                artist.remove()
            if temp_rect_patch:
                temp_rect_patch.remove()

            if len(clicked_points) == 1:
                ax.plot(x, y, 'ro', markersize=2)  # Draw the first click point
                fig.canvas.draw_idle()
                print("First click point drawn.")
            elif len(clicked_points) == 2:
                x_coords = [p[0] for p in clicked_points]
                y_coords = [p[1] for p in clicked_points]

                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                final_roi_coords = (x_min, y_min, x_max, y_max)
                print(f"Final ROI selected: {final_roi_coords}")

                # Draw final rectangle
                temp_rect_patch = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                                fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(temp_rect_patch)
                fig.canvas.draw_idle()
                print("Final ROI rectangle drawn.")

                # Close the figure
                plt.close(fig)
                print("Figure closed.")

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    print("Mouse click event connected. Waiting for clicks...")
    plt.show()

    fig.canvas.mpl_disconnect(cid)
    print("Mouse click event disconnected.")

    if final_roi_coords:
        print(f"select_roi function returns ROI: {final_roi_coords}")
    else:
        print("select_roi function returns None (ROI not selected).")
    return final_roi_coords


# --- Main Program Starts ---
if __name__ == "__main__":
    # 1. User selects Open Field ROI, now taking screenshot from frame 1000
    print("Please draw a rectangle for the open field on the displayed video frame 1000.")
    open_field_roi = select_roi(video_file, frame_to_capture=1000)  # Ensure the correct select_roi is called here

    if open_field_roi is None:
        print("No ROI selected. Exiting.")
        exit()

    of_x_min, of_y_min, of_x_max, of_y_max = open_field_roi
    print(f"\nSelected Open Field ROI: X({of_x_min:.0f}-{of_x_max:.0f}), Y({of_y_min:.0f}-{of_y_max:.0f})")

    # Calculate the center 1/4 area
    of_width = of_x_max - of_x_min
    of_height = of_y_max - of_y_min

    center_x_min = of_x_min + of_width / 4
    center_x_max = of_x_max - of_width / 4
    center_y_min = of_y_min + of_height / 4
    center_y_max = of_y_max - of_height / 4

    # Dynamically define 'zones' dictionary for the center area
    zones = {
        "中心区域": {  # Center Zone
            "x_min": int(center_x_min), "x_max": int(center_x_max),
            "y_min": int(center_y_min), "y_max": int(center_y_max)
        }
    }
    print(f"Calculated center 1/4 area: X({zones['中心区域']['x_min']}-{zones['中心区域']['x_max']}), "
          f"Y({zones['中心区域']['y_min']}-{zones['中心区域']['y_max']})")

    # 2. User defines analysis time range
    while True:
        try:
            analysis_start_time_sec = float(
                input("\n请输入分析开始时间 (秒): "))  # Please enter analysis start time (seconds)
            analysis_end_time_sec = float(
                input("请输入分析结束时间 (秒): "))  # Please enter analysis end time (seconds)
            if analysis_start_time_sec >= 0 and analysis_end_time_sec > analysis_start_time_sec:
                break
            else:
                print(
                    "无效的时间输入。开始时间必须非负，结束时间必须大于开始时间。")  # Invalid time input. Start time must be non-negative, end time must be greater than start time.
        except ValueError:
            print("无效输入。请输入一个数字。")  # Invalid input. Please enter a number.

    try:
        df = pd.read_csv(file_to_analyze, header=[1, 2])
    except FileNotFoundError:
        print(
            f"错误：文件 '{file_to_analyze}' 未找到。请确保文件路径正确。")  # Error: File '{file_to_analyze}' not found. Please ensure the file path is correct.
        exit()
    except Exception as e:
        print(f"读取CSV文件时发生错误：{e}")  # Error reading CSV file: {e}
        print(
            "请检查CSV文件的格式或尝试不同的 header 参数。")  # Please check the CSV file format or try different header parameters.
        try:
            temp_df_header = pd.read_csv(file_to_analyze, nrows=3, header=None)
            print(
                "\nCSV文件前几行的原始内容（用于调试列名）：")  # Original content of the first few lines of the CSV file (for debugging column names):
            print(temp_df_header.to_string(index=False, header=False))
        except:
            pass
        exit()

    print(
        f"\n========== 开始分析文件: {file_to_analyze} ==========")  # ========== Starting analysis of file: {file_to_analyze} ==========
    print(
        f"分析时间段: {analysis_start_time_sec:.2f}秒 到 {analysis_end_time_sec:.2f}秒")  # Analysis period: {analysis_start_time_sec:.2f}s to {analysis_end_time_sec:.2f}s

    # Filter DataFrame by analysis time
    start_frame = int(analysis_start_time_sec * fps)
    end_frame = int(analysis_end_time_sec * fps)
    df_filtered = df.iloc[start_frame:end_frame].copy()

    if df_filtered.empty:
        print(
            "在指定分析时间范围内没有可用数据。退出。")  # No data available within the specified analysis time range. Exiting.
        exit()

    # *** IMPORTANT UPDATE: List all body parts to consider for centroid calculation ***
    # These names correspond to the keys in BODYPART_COLUMN_MAP.
    bodyparts_to_consider_for_each_individual = ['nose', 'bodycenter', 'tailbase']

    # --- Step 1: Get frame-by-frame status for all individuals in the "Center Zone" ---
    # Stores in_zone mask and original frame indices for each individual in the center zone
    ind_zone_statuses = {
        'individual1': {},
        'individual2': {},
        'individual3': {}
    }

    # Iterate through each individual and the single "Center Zone"
    for ind_prefix in ind_zone_statuses.keys():
        for zone_name, details in zones.items():  # This will iterate only once for "中心区域"
            x_coords, y_coords, in_zone_mask, original_frames = get_individual_zone_status(
                df_filtered, ind_prefix, details, bodyparts_to_consider_for_each_individual, likelihood_threshold
            )
            if not in_zone_mask.empty:
                ind_zone_statuses[ind_prefix][zone_name] = {
                    'in_zone_mask': in_zone_mask,
                    'original_frames': original_frames
                }

    # --- Step 2: Apply exclusivity logic and calculate dwelling time ---
    all_individuals = ['individual1', 'individual2', 'individual3']
    all_total_times = {ind: {} for ind in all_individuals}

    # 1) Build "occupancy table" for each zone: rows = original frame indices, columns = individuals,
    #    values = whether the individual is in that zone for that frame
    zone_occupancy = {}
    for zone_name in zones:  # This will run only for "中心区域"
        # The index should be aligned with df_filtered's index, as we are only concerned with frames within df_filtered
        occ_df = pd.DataFrame(False, index=df_filtered.index, columns=all_individuals)
        for ind in all_individuals:
            if zone_name in ind_zone_statuses[ind]:
                frames = ind_zone_statuses[ind][zone_name]['original_frames']  # This is a list
                mask = ind_zone_statuses[ind][zone_name]['in_zone_mask']  # This is a pandas Series

                # Convert frames (list) to Series to use isin()
                frames_series = pd.Series(frames)
                # Filter frames that are both in frames_series and in occ_df's index
                valid_frames_in_occ_df = frames_series[frames_series.isin(occ_df.index)]

                # Use loc and reindex to ensure mask alignment.
                # The original mask's index is already the original frame indices, so we can directly use .loc
                # Only assign values for indices present in both valid_frames_in_occ_df and mask.index
                occ_df.loc[valid_frames_in_occ_df, ind] = mask.loc[valid_frames_in_occ_df].values

        zone_occupancy[zone_name] = occ_df  # This occ_df now contains only frames within the filtered time period

    # 2) Calculate dwelling time after exclusivity for each individual and zone
    for ind in all_individuals:
        for zone_name, details in zones.items():  # This will run only for "中心区域"
            # If no data for this individual in this zone within the filtered time period, set to zero
            if zone_name not in ind_zone_statuses[ind]:
                all_total_times[ind][zone_name] = 0.0
                continue

            raw_mask = ind_zone_statuses[ind][zone_name]['in_zone_mask']
            raw_frames = ind_zone_statuses[ind][zone_name]['original_frames']

            occ_df_for_zone = zone_occupancy[
                zone_name]  # This occupancy table already contains frames within the filtered time period

            # Construct the mask for "only the current individual occupies the zone in this frame"
            # Condition 1: Only 1 animal in the zone for this frame -> occ_df_for_zone.sum(axis=1)==1
            # Condition 2: The current individual is in the zone for this frame -> occ_df_for_zone[ind]==True
            exclusive_full = (occ_df_for_zone.sum(axis=1) == 1) & (occ_df_for_zone[ind])

            # Map back to the "cleaned" frame sequence (from raw_frames)
            # Both exclusive_full and raw_frames should be within the same filtered time period
            final_mask_series = exclusive_full.reindex(raw_frames)
            final_mask = final_mask_series.fillna(False)  # Fill NaNs (frames not in exclusive_full) with False

            _, total_time = calculate_zone_dwelling_time(
                final_mask,
                raw_frames,
                fps,
                ind,
                zone_name
            )
            all_total_times[ind][zone_name] = total_time

    # Finally, print summary
    print("\n===== 汇总结果 =====")  # ===== Summary Results =====
    for ind in all_individuals:
        for zone_name in zones:
            print(
                f"{ind} 在 {zone_name} 总停留时间: {all_total_times[ind][zone_name]:.2f} 秒")  # {ind} total dwelling time in {zone_name}: {total_time:.2f} seconds

    # Overall summary
    print(
        f"\n\n========== 整体汇总 (文件: {file_to_analyze}, 使用重心点) ==========")  # ========== Overall Summary (File: {file_to_analyze}, using centroid) ==========
    for zone_name in zones.keys():
        print(
            f"个体 1 在 {zone_name} 总停留时间: {all_total_times['individual1'].get(zone_name, 0.0):.2f} 秒")  # Individual 1 total dwelling time in {zone_name}: {total_time:.2f} seconds
        print(
            f"个体 2 在 {zone_name} 总停留时间: {all_total_times['individual2'].get(zone_name, 0.0):.2f} 秒")  # Individual 2 total dwelling time in {zone_name}: {total_time:.2f} seconds
        print(
            f"个体 3 在 {zone_name} (排他性条件) 总停留时间: {all_total_times['individual3'].get(zone_name, 0.0):.2f} 秒")  # Individual 3 total dwelling time in {zone_name} (exclusive condition): {total_time:.2f} seconds
