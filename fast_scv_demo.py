import os
import json
import shutil
import argparse
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# mediapipe 기준 21개 landmark에 대해 tactile 센서 값 매핑 (단순 1:1 매핑으로 사용)
mediapipe_tactile_groups = {f"LM{i}": [i] for i in range(21)}

def convert_tactile_to_list(tactile_data):
    """
    tactile.json의 데이터를 읽어, 각 프레임마다 좌측/우측 tactile 센서 데이터를 추출합니다.
    각 센서는 4x4x3 배열로 주어지며, 이를 (24,4,3) 배열로 reshape합니다.
    """
    right_sensor_ids = ['128', '129', '130', '131', '132', '133']
    left_sensor_ids  = ['134', '135', '136', '137', '138', '139']
    
    init_tactile_data = {}
    for entry in tactile_data:
        entry = entry['tactile']
        for right_id in right_sensor_ids:
            if right_id in entry:
                init_tactile_data[right_id] = entry[right_id]
        for left_id in left_sensor_ids:
            if left_id in entry:
                init_tactile_data[left_id] = entry[left_id]
    
    # 예시: 센서 '129'가 없으면 '128'과 동일한 크기의 0 배열로 초기화
    init_tactile_data['129'] = {'data': np.full_like(tactile_data[10]['tactile']['128']['data'], 0)}
    
    for entry in tactile_data:
        entry = entry['tactile']
        for right_id in right_sensor_ids:
            if right_id not in entry:
                entry[right_id] = init_tactile_data[right_id]
        for left_id in left_sensor_ids:
            if left_id not in entry:
                entry[left_id] = init_tactile_data[left_id]
                
    tactile_left_list, tactile_right_list = [], []
    for entry in tactile_data:
        entry = entry['tactile']
        left_arrays = []
        right_arrays = []
        for left_id in left_sensor_ids:
            sensor_value = np.array(entry[left_id]['data']).reshape(4, 4, 3)
            init_value = np.array(init_tactile_data[left_id]['data']).reshape(4, 4, 3)
            sensor_value = sensor_value - init_value
            left_arrays.append(sensor_value)
        for right_id in right_sensor_ids:
            sensor_value = np.array(entry[right_id]['data']).reshape(4, 4, 3)
            sensor_value = sensor_value - init_value  # 단순화를 위해 동일한 초기값 사용
            right_arrays.append(sensor_value)
        left_arr = np.stack(left_arrays, axis=0).reshape(24, 4, 3)
        right_arr = np.stack(right_arrays, axis=0).reshape(24, 4, 3)
        tactile_left_list.append(left_arr)
        tactile_right_list.append(right_arr)
    return tactile_left_list, tactile_right_list

def merge_tactile_lists(tactile_left_list, tactile_right_list):
    """
    좌측과 우측 tactile 데이터를 프레임별로 element-wise 평균하여 결합합니다.
    """
    merged = []
    n_frames = min(len(tactile_left_list), len(tactile_right_list))
    for i in range(n_frames):
        merged.append((tactile_left_list[i] + tactile_right_list[i]) / 2.0)
    return merged

def get_tactile_values(tactile_array, tactile_norm=True):
    """
    tactile_array의 shape은 (24, 4, 3)입니다.
    tactile_norm=True이면 각 센서값의 평균을 계산하여 24개 리스트를 반환하고,
    mediapipe는 21개 landmark를 사용하므로 앞의 21개 값만 반환합니다.
    """
    if tactile_norm:
        tactile_values = tactile_array.mean(axis=(1,2)).tolist()
        return tactile_values[:21]
    else:
        tactile_values_x = tactile_array[:, :, 0].mean(axis=1).tolist()
        tactile_values_y = tactile_array[:, :, 1].mean(axis=1).tolist()
        tactile_values_z = tactile_array[:, :, 2].mean(axis=1).tolist()
        return tactile_values_x[:21], tactile_values_y[:21], tactile_values_z[:21]

def draw_tactile_overlay_on_frame(frame, hand_landmarks, tactile_values, tactile_threshold):
    """
    주어진 hand_landmarks(mediapipe 결과)와 tactile 센서값을 이용해,
    각 관절(landmark) 위치에 원(circle)을 그려 센서값을 오버레이합니다.
    센서값을 tactile_threshold로 나눈 후 [0,1] 범위로 클립합니다.
    """
    h, w, _ = frame.shape
    points = []
    for lm in hand_landmarks.landmark:
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append((x, y))
    
    # 스켈레톤 연결선 그리기 (mediapipe 기본 연결 정보 사용)
    connections = mp.solutions.hands.HAND_CONNECTIONS
    for conn in connections:
        start_idx, end_idx = conn
        if start_idx < len(points) and end_idx < len(points):
            cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)
    
    # 각 관절에 tactile 센서 오버레이 그리기
    for i, pt in enumerate(points):
        sensor_val = tactile_values[i] if i < len(tactile_values) else 0
        # 센서값을 tactile_threshold로 정규화하고, 음수는 0으로 클립
        norm_val = sensor_val / tactile_threshold
        norm_val = np.clip(norm_val, 0, 1)
        # 원의 반지름: 최소 5, 최대 15
        radius = int(5 + norm_val * 10)
        cmap = plt.get_cmap('jet')
        color_rgba = cmap(norm_val)
        # matplotlib colormap은 RGBA; OpenCV는 BGR, so convert accordingly.
        color = (int(color_rgba[2]*255), int(color_rgba[1]*255), int(color_rgba[0]*255))
        if radius > 0:
            cv2.circle(frame, pt, radius, color, -1)
    return frame

def process_video_mediapipe(video_path, output_video_path, merged_tactile_list, tactile_threshold, tactile_norm):
    """
    left_video.mp4 파일을 읽어 mediapipe로 손 landmark를 추출하고,
    각 프레임마다 tactile 센서 오버레이를 그린 후 결과 영상을 저장합니다.
    """
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=2,
                           min_detection_confidence=0.5)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # tactile 센서값 처리: 프레임 인덱스에 맞게 merge된 tactile 데이터를 가져옴
        if frame_idx < len(merged_tactile_list):
            tactile_array = merged_tactile_list[frame_idx]
        else:
            tactile_array = merged_tactile_list[-1]
        tactile_values = get_tactile_values(tactile_array, tactile_norm)

        # mediapipe 처리: RGB 변환 후 손 landmark 검출
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                frame = draw_tactile_overlay_on_frame(frame, hand_landmarks, tactile_values, tactile_threshold)
        writer.write(frame)
        frame_idx += 1
    cap.release()
    writer.release()
    print(f"Processed: {video_path} -> {output_video_path}")

def process_episode(episode_dir_path, args_dict):
    """
    episode 폴더 내의 left_video.mp4와 tactile.json 파일을 읽어 영상 처리 후,
    결과물을 output_dir/episode_name 폴더에 저장합니다.
    """
    args = argparse.Namespace(**args_dict)
    episode_dir = Path(episode_dir_path)
    video_path = episode_dir / 'left_video.mp4'
    tactile_json_path = episode_dir / 'tactile.json'
    if not (video_path.exists() and tactile_json_path.exists()):
        print(f"Missing required files in {episode_dir.name}. Skipping.")
        return
    output_episode_dir = Path(args.output_dir) / episode_dir.name
    os.makedirs(output_episode_dir, exist_ok=True)
    
    # tactile.json 파일 로드
    with open(tactile_json_path, 'r') as f:
        tactile_data_all = json.load(f)
    tactile_left_list, tactile_right_list = convert_tactile_to_list(tactile_data_all)
    merged_tactile_list = merge_tactile_lists(tactile_left_list, tactile_right_list)
    
    output_video = output_episode_dir / 'left_video.mp4'
    process_video_mediapipe(str(video_path), str(output_video), merged_tactile_list, args.tactile_threshold, args.tactile_norm)
    
    # tactile.json 파일도 결과 폴더에 복사
    shutil.copy(str(tactile_json_path), str(output_episode_dir / 'tactile.json'))
    print(f"Episode {episode_dir.name} processed.")

def main():
    parser = argparse.ArgumentParser(
        description='Mediapipe 기반 양손 스켈레톤 및 촉각 센서값 오버레이 데모 (left_video.mp4 입력)'
    )
    parser.add_argument('--raw_dataset', type=str, default='raw_dataset',
                        help='episode 디렉토리를 포함한 입력 폴더 경로')
    parser.add_argument('--output_dir', type=str, default='output_dir',
                        help='합성 결과물을 저장할 출력 폴더')
    parser.add_argument('--tactile_threshold', type=float, default=4,
                        help='tactile 값 임계치 설정')
    parser.add_argument('--tactile_norm', dest='tactile_norm', action='store_true', default=True,
                        help='tactile 값 정규화 활성화 (False면 각 축별로 따로 저장)')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='에피소드 병렬 처리시 사용할 프로세스 수')
    args = parser.parse_args()
    
    raw_dataset_path = Path(args.raw_dataset)
    episode_dirs = [str(ep) for ep in sorted(raw_dataset_path.iterdir()) if ep.is_dir()]
    args_dict = vars(args)
    for ep in episode_dirs:
        process_episode(ep, args_dict)

if __name__ == '__main__':
    main()
