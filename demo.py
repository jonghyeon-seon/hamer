from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full

LIGHT_BLUE = (0.65098039,  0.74117647,  0.85882353)

from vitpose_model import ViTPoseModel

import json
from typing import Dict, Optional

def main():
    parser = argparse.ArgumentParser(description='HaMeR demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--img_folder', type=str, default='images', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='out_demo', help='Output folder to save rendered results')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=True, help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--body_detector', type=str, default='vitdet', choices=['vitdet', 'regnety'], help='Using regnety improves runtime and reduces memory')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider')
    # overlay_type 인자 수정: 'joint', 'skin_point', 'skin_mesh'
    parser.add_argument('--overlay_type', type=str, default='joint', choices=['joint', 'skin_point', 'skin_mesh'], help='Overlay 방식을 선택 (joint: skeleton joint 기반, skin_point: point 표시, skin_mesh: 기존 face 기반)')
    args = parser.parse_args()

    # Download and load checkpoints
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)

    # Setup HaMeR model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Load detector
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.body_detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hamer
        cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
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
        detector       = DefaultPredictor_Lazy(detectron2_cfg)

    # keypoint detector
    cpm = ViTPoseModel(device)

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    # tactile 영역을 위한 vertex 그룹 정의 (총 24 그룹)
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

    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)

    # Get all demo images ends with .jpg or .png
    img_paths = [img for end in args.file_type for img in Path(args.img_folder).glob(end)]

    # Iterate over all images in folder
    for img_path in img_paths:
        img_cv2 = cv2.imread(str(img_path))

        # Detect humans in image
        det_out = detector(img_cv2)
        img = img_cv2.copy()[:, :, ::-1]

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()

        # Detect human keypoints for each person
        vitposes_out = cpm.predict_pose(
            img,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        bboxes = []
        is_right = []

        # Use hands based on hand keypoint detections
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            # Rejecting not confident detections
            keyp = left_hand_keyp
            valid = keyp[:, 2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
                bboxes.append(bbox)
                is_right.append(0)
            keyp = right_hand_keyp
            valid = keyp[:, 2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
                bboxes.append(bbox)
                is_right.append(1)

        if len(bboxes) == 0:
            continue

        boxes = np.stack(bboxes)
        right = np.stack(is_right)

        # Run reconstruction on all detected hands
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []
        all_tactile_values = []   # tactile 값을 저장할 리스트 추가
        all_joint_keypoints = []  # skeleton joint 좌표 저장 리스트 (각각 np.array, shape: (21,3))

        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            multiplier = (2 * batch['right'] - 1)
            pred_cam = out['pred_cam']
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                img_fn, _ = os.path.splitext(os.path.basename(img_path))
                person_id = int(batch['personid'][n])
                white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:, None, None] / 255) / (DEFAULT_STD[:, None, None] / 255)
                input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:, None, None] / 255) + (DEFAULT_MEAN[:, None, None] / 255)
                input_patch_np = input_patch.permute(1, 2, 0).numpy()
                regression_img = renderer(
                    out['pred_vertices'][n].detach().cpu().numpy(),
                    out['pred_cam_t'][n].detach().cpu().numpy(),
                    batch['img'][n],
                    mesh_base_color=LIGHT_BLUE,
                    scene_bg_color=(1, 1, 1),
                    is_right=batch['right'][n].cpu().numpy()
                )

                # 단일 렌더링용 sensor overlay (데모용 랜덤 값)
                tactile_values = np.random.rand(24).tolist()
                tactile_values = np.zeros(24)
                # tactile_values[13] = 10
                # tactile_values[14] = 10
                # tactile_values[15] = 10
                tactile_values[23] = 10
                is_right_flag = batch['right'][n].cpu().numpy()
                # 만약 overlay_type이 'joint'라면, skeleton joint keypoints를 전달 (21개)
                joint_keypoints = None
                if args.overlay_type == 'joint':
                    joint_keypoints = out['pred_keypoints_3d'][n].detach().cpu().numpy()
                tactile_img = renderer(
                    out['pred_vertices'][n].detach().cpu().numpy(),
                    out['pred_cam_t'][n].detach().cpu().numpy(),
                    batch['img'][n],
                    mesh_base_color=LIGHT_BLUE,
                    scene_bg_color=(1, 1, 1),
                    tactile_values=tactile_values,
                    tactile_vertex_groups=tactile_vertex_groups,
                    tactile_opacity=1.0,
                    mesh_alpha=0.1,
                    is_right=is_right_flag,
                    overlay_type=args.overlay_type,        # 'joint', 'skin_point' 또는 'skin_mesh'
                    joint_sphere_radius=0.02,                 # joint sphere의 반지름 (필요시 조정)
                    joint_alpha=1.0,
                    joint_cmap='Reds',
                    joint_keypoints_3d=joint_keypoints,
                    tactile_sensor_threshold=0
                )

                if args.side_view:
                    side_img = renderer(
                        out['pred_vertices'][n].detach().cpu().numpy(),
                        out['pred_cam_t'][n].detach().cpu().numpy(),
                        white_img,
                        mesh_base_color=LIGHT_BLUE,
                        scene_bg_color=(1, 1, 1),
                        side_view=True,
                        tactile_values=tactile_values,
                        tactile_vertex_groups=tactile_vertex_groups,
                        tactile_opacity=1.0,
                        mesh_alpha=0.1,
                        is_right=is_right_flag,
                        overlay_type=args.overlay_type,
                        joint_sphere_radius=0.02,
                        joint_alpha=1.0,
                        joint_cmap='Reds',
                        joint_keypoints_3d=joint_keypoints,
                        tactile_sensor_threshold=0
                    )

                    final_img = np.concatenate([input_patch_np, regression_img, tactile_img, side_img], axis=1)
                else:
                    final_img = np.concatenate([input_patch_np, regression_img, tactile_img], axis=1)

                cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_{person_id}.png'), 255 * final_img[:, :, ::-1])

                verts = out['pred_vertices'][n].detach().cpu().numpy()
                cam_t = pred_cam_t_full[n]
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right_flag)
                all_tactile_values.append(tactile_values)
                all_joint_keypoints.append(joint_keypoints)

                if args.save_mesh:
                    camera_translation = cam_t.copy()
                    tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_BLUE, is_right=is_right_flag)
                    tmesh.export(os.path.join(args.out_folder, f'{img_fn}_{person_id}.obj'))

        # full_frame 렌더링 (모든 사람) – overlay 포함
        if args.full_frame and len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
                tactile_values_list=all_tactile_values,
                tactile_vertex_groups=tactile_vertex_groups,
                tactile_opacity=1.0,
                tactile_cmap='Reds',
                mesh_alpha=0.3,  # 메인 손 메쉬를 반투명하게 설정
                overlay_type=args.overlay_type,
                joint_sphere_radius=0.02,
                joint_alpha=1.0,
                joint_cmap='Reds',
                joint_keypoints_3d_list=all_joint_keypoints,
                tactile_sensor_threshold=0
            )
            cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n],
                                                     is_right=all_right, **misc_args)

            input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)
            input_img_overlay = input_img[:, :, :3] * (1 - cam_view[:, :, 3:4]) + cam_view[:, :, :3] * cam_view[:, :, 3:4]
            print(f'{img_fn}_all.jpg')
            cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_all.jpg'), 255 * input_img_overlay[:, :, ::-1])

if __name__ == '__main__':
    main()
