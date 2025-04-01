import os
import json
import shutil
from pathlib import Path
import argparse
import torch
import cv2
import numpy as np
from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full
from vitpose_model import ViTPoseModel

# tactile 영역을 위한 vertex 그룹 정의 (총 24 그룹)
LIGHT_BLUE = (0.65098039,  0.74117647,  0.85882353)
tactile_vertex_groups = {
    "Thumb1": [739, 740, 741, 755, 756, 757, 758, 759, 760, 761, 762, 763],
    "Thumb2": [31, 124, 125, 267, 698, 699, 700, 701, 704],
    "Thumb3": [7, 9, 240, 123, 126, 266],
    "Index1": [321, 322, 323, 325, 326, 328, 329, 330, 331, 342, 343, 344, 347, 349, 350],
    "Index2": [46, 47, 155, 223, 224, 237, 238, 245, 280, 281],
    "Index3": [137, 139, 140, 164, 165, 170, 171, 173, 189, 194, 195, 212],
    "Middle1": [418, 432, 433, 435, 436, 438, 439, 440, 441, 449, 454, 455, 456, 459, 461, 462],
    "Middle2": [356, 357, 372, 396, 397, 398, 402, 403],
    "Middle3": [370, 371, 374, 375, 378, 379, 380, 385, 386, 387],
    "Ring1": [523, 546, 547, 549, 550, 551, 552, 565, 566, 567, 568, 569, 570, 571, 572, 573],
    "Ring2": [468, 469, 502, 503, 506, 507, 513, 514, 516],
    "Ring3": [484, 485, 488, 489, 496, 497, 510, 579],
    "Little1": [666, 667, 668, 682, 683, 684, 685, 686, 687, 688, 689, 690, 663, 664, 669],
    "Little2": [580, 581, 598, 620, 621, 624, 625, 626, 630, 631],
    "Little3": [596, 597, 600, 601, 606, 607, 608, 613, 614, 615],
    "Palm1": [102, 278, 594, 595, 604, 605, 769, 770, 771, 775, 776],
    "Palm2": [76, 77, 141, 142, 147, 148, 196, 197, 275],
    "Palm3": [74, 75, 151, 152, 228, 268, 271, 288],
    "Palm4": [62, 63, 64, 65, 93, 132, 138, 149, 150, 168, 169],
    "Palm5": [70, 71, 72, 73, 157, 159, 188, 777],
    "Palm6": [24, 27, 66, 67, 68, 69],
    "Palm7": [32, 45, 130, 131, 243, 244, 255],
    "Palm8": [25, 109, 111, 112, 264, 265, 285],
    "Palm9": [1, 2, 4, 7, 9, 113, 115, 240]
}

# 수정된 process_video 함수:
def process_video(video_path, output_video_path, tactile_left_list, tactile_right_list,
                  model, model_cfg, detector, cpm, renderer, args, device):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"영상 {video_path}을 열 수 없습니다.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    if args.tactile_norm:
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    else:
        base, ext = os.path.splitext(output_video_path)
        writer_x = cv2.VideoWriter(base + "_x" + ext, fourcc, fps, (width, height))
        writer_y = cv2.VideoWriter(base + "_y" + ext, fourcc, fps, (width, height))
        writer_z = cv2.VideoWriter(base + "_z" + ext, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_cv2 = frame.copy()
        det_out = detector(img_cv2)
        img_rgb = img_cv2[:, :, ::-1].copy()  # BGR -> RGB
        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        if valid_idx.sum() == 0:
            if args.tactile_norm:
                writer.write(frame)
            else:
                writer_x.write(frame)
                writer_y.write(frame)
                writer_z.write(frame)
            frame_idx += 1
            continue

        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()

        vitposes_out = cpm.predict_pose(
            img_rgb,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)]
        )

        bboxes = []
        is_right_list = []
        # 양손 모두에 대해 검출 수행
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]
            # 왼손 검출
            valid_left = (left_hand_keyp[:, 2] > 0.5)
            if valid_left.sum() > 3:
                bbox_left = [left_hand_keyp[valid_left, 0].min(), left_hand_keyp[valid_left, 1].min(),
                             left_hand_keyp[valid_left, 0].max(), left_hand_keyp[valid_left, 1].max()]
                bboxes.append(bbox_left)
                is_right_list.append(0)
            # 오른손 검출
            valid_right = (right_hand_keyp[:, 2] > 0.5)
            if valid_right.sum() > 3:
                bbox_right = [right_hand_keyp[valid_right, 0].min(), right_hand_keyp[valid_right, 1].min(),
                              right_hand_keyp[valid_right, 0].max(), right_hand_keyp[valid_right, 1].max()]
                bboxes.append(bbox_right)
                is_right_list.append(1)

        if len(bboxes) == 0:
            if args.tactile_norm:
                writer.write(frame)
            else:
                writer_x.write(frame)
                writer_y.write(frame)
                writer_z.write(frame)
            frame_idx += 1
            continue

        boxes = np.stack(bboxes)
        right_arr = np.array(is_right_list)

        # 각 프레임마다 왼손, 오른손 tactile 데이터를 가져옵니다.
        if frame_idx < len(tactile_left_list):
            tactile_left_reading = tactile_left_list[frame_idx]
        else:
            tactile_left_reading = tactile_left_list[-1]
        if frame_idx < len(tactile_right_list):
            tactile_right_reading = tactile_right_list[frame_idx]
        else:
            tactile_right_reading = tactile_right_list[-1]

        # tactile 값은 각 검출(손)별로 할당합니다.
        if args.tactile_norm:
            all_tactile_values = []
        else:
            all_tactile_values_x = []
            all_tactile_values_y = []
            all_tactile_values_z = []

        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right_arr, rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(bboxes), shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []
        all_joint_keypoints = []

        # 미리 tactile 데이터의 평균값 계산 (정규화 여부에 따라)
        if args.tactile_norm:
            tactile_left_values = tactile_left_reading.mean(axis=(1,2)).tolist()
            tactile_right_values = tactile_right_reading.mean(axis=(1,2)).tolist()
        else:
            tactile_left_values_x = tactile_left_reading[:, :, 0].mean(axis=1).tolist()
            tactile_left_values_y = tactile_left_reading[:, :, 1].mean(axis=1).tolist()
            tactile_left_values_z = tactile_left_reading[:, :, 2].mean(axis=1).tolist()
            tactile_right_values_x = tactile_right_reading[:, :, 0].mean(axis=1).tolist()
            tactile_right_values_y = tactile_right_reading[:, :, 1].mean(axis=1).tolist()
            tactile_right_values_z = tactile_right_reading[:, :, 2].mean(axis=1).tolist()

        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out_dict = model(batch)
            multiplier = (2 * batch['right'] - 1)
            pred_cam = out_dict['pred_cam']
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            box_center = batch["box_center"].float()
            box_size   = batch["box_size"].float()
            img_size_tensor = batch["img_size"].float()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size_tensor.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size_tensor, scaled_focal_length).detach().cpu().numpy()
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                verts = out_dict['pred_vertices'][n].detach().cpu().numpy()
                cam_t = pred_cam_t_full[n]
                is_right_flag = batch['right'][n].cpu().numpy()
                joint_keypoints = None
                if args.overlay_type == 'joint':
                    joint_keypoints = out_dict['pred_keypoints_3d'][n].detach().cpu().numpy()
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right_flag)
                if args.tactile_norm:
                    # 손 타입에 따라 해당 tactile 값을 할당합니다.
                    if is_right_flag:
                        all_tactile_values.append(tactile_right_values)
                    else:
                        all_tactile_values.append(tactile_left_values)
                else:
                    if is_right_flag:
                        all_tactile_values_x.append(tactile_right_values_x)
                        all_tactile_values_y.append(tactile_right_values_y)
                        all_tactile_values_z.append(tactile_right_values_z)
                    else:
                        all_tactile_values_x.append(tactile_left_values_x)
                        all_tactile_values_y.append(tactile_left_values_y)
                        all_tactile_values_z.append(tactile_left_values_z)
                all_joint_keypoints.append(joint_keypoints)

        if args.full_frame and len(all_verts) > 0:
            misc_args_common = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
                tactile_vertex_groups=tactile_vertex_groups,
                tactile_opacity=1.0,
                tactile_cmap='Reds',
                mesh_alpha=0.3,
                overlay_type=args.overlay_type,
                joint_sphere_radius=0.02,
                joint_alpha=1.0,
                joint_cmap='Reds',
                joint_keypoints_3d_list=all_joint_keypoints,
                tactile_sensor_threshold=0
            )
            render_res = (width, height)
            input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0  # BGR -> RGB
            input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)
            if args.tactile_norm:
                misc_args = misc_args_common.copy()
                misc_args['tactile_values_list'] = all_tactile_values
                cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=render_res,
                                                         is_right=all_right, **misc_args)
                if args.with_bg:
                    input_img_overlay = input_img[:, :, :3] * (1 - cam_view[:, :, 3:4]) + cam_view[:, :, :3] * cam_view[:, :, 3:4]
                else:
                    input_img_overlay = cam_view[:, :, :3]
                final_frame = (255 * input_img_overlay[:, :, ::-1]).astype(np.uint8)
                writer.write(final_frame)
            else:
                misc_args_x = misc_args_common.copy()
                misc_args_y = misc_args_common.copy()
                misc_args_z = misc_args_common.copy()
                misc_args_x['tactile_values_list'] = all_tactile_values_x
                misc_args_y['tactile_values_list'] = all_tactile_values_y
                misc_args_z['tactile_values_list'] = all_tactile_values_z
                cam_view_x = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=render_res,
                                                            is_right=all_right, **misc_args_x)
                cam_view_y = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=render_res,
                                                            is_right=all_right, **misc_args_y)
                cam_view_z = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=render_res,
                                                            is_right=all_right, **misc_args_z)
                if args.with_bg:
                    input_img_overlay_x = input_img[:, :, :3] * (1 - cam_view_x[:, :, 3:4]) + cam_view_x[:, :, :3] * cam_view_x[:, :, 3:4]
                    input_img_overlay_y = input_img[:, :, :3] * (1 - cam_view_y[:, :, 3:4]) + cam_view_y[:, :, :3] * cam_view_y[:, :, 3:4]
                    input_img_overlay_z = input_img[:, :, :3] * (1 - cam_view_z[:, :, 3:4]) + cam_view_z[:, :, :3] * cam_view_z[:, :, 3:4]
                else:
                    input_img_overlay_x = cam_view_x[:, :, :3]
                    input_img_overlay_y = cam_view_y[:, :, :3]
                    input_img_overlay_z = cam_view_z[:, :, :3]
                final_frame_x = (255 * input_img_overlay_x[:, :, ::-1]).astype(np.uint8)
                final_frame_y = (255 * input_img_overlay_y[:, :, ::-1]).astype(np.uint8)
                final_frame_z = (255 * input_img_overlay_z[:, :, ::-1]).astype(np.uint8)
                writer_x.write(final_frame_x)
                writer_y.write(final_frame_y)
                writer_z.write(final_frame_z)
        else:
            if args.tactile_norm:
                writer.write(frame)
            else:
                writer_x.write(frame)
                writer_y.write(frame)
                writer_z.write(frame)

        frame_idx += 1

    cap.release()
    if args.tactile_norm:
        writer.release()
    else:
        writer_x.release()
        writer_y.release()
        writer_z.release()
    print(f"처리 완료: {video_path} -> {output_video_path} (tactile_norm={args.tactile_norm})")


# 에피소드 단위 처리 함수 수정 (left_video.mp4만 사용)
def process_episode(episode_dir_path, args_dict):
    args = argparse.Namespace(**args_dict)
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # body detector 설정
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.body_detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hamer
        cfg_path = Path(hamer.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    elif args.body_detector == 'regnety':
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
        detector = DefaultPredictor_Lazy(detectron2_cfg)

    cpm = ViTPoseModel(device)
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    episode_dir = Path(episode_dir_path)
    left_video_path = episode_dir / 'left_video.mp4'
    tactile_json_path = episode_dir / 'tactile.json'

    if not (left_video_path.exists() and tactile_json_path.exists()):
        print(f"{episode_dir.name}에 필수 파일이 없습니다. 건너뜁니다.")
        return

    output_episode_dir = Path(args.output_dir) / episode_dir.name
    os.makedirs(output_episode_dir, exist_ok=True)

    # tactile.json 로드 및 왼손/오른손 tactile 데이터 분리
    with open(tactile_json_path, 'r') as f:
        tactile_data_all = json.load(f)
        tactile_left_list, tactile_right_list = convert_tactile_to_list(tactile_data_all)

    output_video = output_episode_dir / 'left_video.mp4'
    process_video(str(left_video_path), str(output_video), tactile_left_list, tactile_right_list,
                  model, model_cfg, detector, cpm, renderer, args, device)

    shutil.copy(str(tactile_json_path), str(output_episode_dir / 'tactile.json'))
    print(f"에피소드 {episode_dir.name} 처리 완료.")


def convert_tactile_to_list(tactile_data):    #converting sequence of sensor's id to sequence of tactile values
    CONVERT_ID_GLOVE_TO_RENDER ={
        0: 4,
        1: 3,
        2: 2,
        3: 1,
        4: 8,
        5: 7,
        6: 6,
        7: 5,
        8: 12,
        9: 11,
        10: 10,
        11: 9,
        12: 16,
        13: 15,
        14: 14,
        15: 13,
        16: 20,
        17: 19,
        18: 18,
        19: 17,
        20: 22,
        21: 0,
        22: 21,
        23: 23,
    }
    right_sensor_ids = ['128', '129', '130', '131', '132', '133']
    left_sensor_ids  = ['134', '135', '136', '137', '138', '139']
    
    init_tactile_data = {}
    for entry in tactile_data:
        entry = entry['tactile']
        for right_id in right_sensor_ids:
            if (right_id in entry) and (right_id not in init_tactile_data):
                init_tactile_data[right_id] = entry[right_id]
        for left_id in left_sensor_ids:
            if (left_id in entry) and (left_id not in init_tactile_data):
                init_tactile_data[left_id] = entry[left_id]
        if len(init_tactile_data) == len(right_sensor_ids) + len(left_sensor_ids):
            break

    if len(init_tactile_data) != len(right_sensor_ids) + len(left_sensor_ids):
        raise ValueError("init tactile data not found")

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
            # sensor_value = sensor_value - init_value
            left_arrays.append(sensor_value)
        for right_id in right_sensor_ids:
            sensor_value = np.array(entry[right_id]['data']).reshape(4, 4, 3)
            # sensor_value = sensor_value - init_value
            right_arrays.append(sensor_value)

        left_arr = np.stack(left_arrays, axis=0).reshape(24, 4, 3)
        right_arr = np.stack(right_arrays, axis=0).reshape(24, 4, 3)
        
        ordered_indices = [k for k, v in sorted(CONVERT_ID_GLOVE_TO_RENDER.items(), key=lambda item: item[1])]
        left_arr = left_arr[ordered_indices]
        right_arr = right_arr[ordered_indices]
        
        tactile_left_list.append(left_arr)
        tactile_right_list.append(right_arr)
        


    return tactile_left_list, tactile_right_list


def main():
    parser = argparse.ArgumentParser(
        description='HaMeR + StateCollectordeVice 데모 (영상 및 tactile 입력 버전, 병렬 처리 지원)'
    )
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT,
                        help='Pretrained model checkpoint 경로')
    parser.add_argument('--raw_dataset', type=str, default='raw_dataset',
                        help='episode 디렉토리를 포함한 입력 폴더 경로')
    parser.add_argument('--output_dir', type=str, default='output_dir',
                        help='합성 결과물을 저장할 출력 폴더')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False,
                        help='Side view 렌더링 활성화')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=True,
                        help='전체 사람을 함께 렌더링')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False,
                        help='메쉬도 저장')
    parser.add_argument('--batch_size', type=int, default=1, help='Inference/fitting 배치 크기')
    parser.add_argument('--rescale_factor', type=float, default=2.0,
                        help='BBox 패딩에 사용할 스케일 팩터')
    parser.add_argument('--body_detector', type=str, default='vitdet', choices=['vitdet', 'regnety'],
                        help='body_detector 선택 (regnety: 속도/메모리 절약)')
    parser.add_argument('--overlay_type', type=str, default='joint',
                        choices=['joint', 'skin_point', 'skin_mesh'],
                        help='Overlay 방식 선택 (joint: skeleton joint 기반 등)')
    parser.add_argument('--with_bg', dest='with_bg', action='store_true', default=False,
                        help='배경 렌더링 활성화')
    parser.add_argument('--tactile_threshold', type=float, default=5.0,
                        help='tactile 값 임계치 설정')
    parser.add_argument('--tactile_norm', dest='tactile_norm', action='store_true', default=True,
                        help='tactile 값 정규화 활성화 (False면 각 축별로 따로 저장)')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='에피소드 병렬 처리시 사용할 프로세스 수')
    args = parser.parse_args()

    raw_dataset_path = Path(args.raw_dataset)
    episode_dirs = [str(ep) for ep in sorted(raw_dataset_path.iterdir()) if ep.is_dir()]

    args_dict = vars(args)

    # 기존의 병렬 처리 대신 순차적으로 처리
    for ep in episode_dirs:
        process_episode(ep, args_dict)


if __name__ == '__main__':
    main()
