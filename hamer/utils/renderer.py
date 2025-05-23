import os
if 'PYOPENGL_PLATFORM' not in os.environ:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
import torch
import numpy as np
import pyrender
import trimesh
import cv2
from yacs.config import CfgNode
from typing import List, Optional

def cam_crop_to_full(cam_bbox, box_center, box_size, img_size, focal_length=5000.):
    # Convert cam_bbox to full image
    img_w, img_h = img_size[:, 0], img_size[:, 1]
    cx, cy, b = box_center[:, 0], box_center[:, 1], box_size
    w_2, h_2 = img_w / 2., img_h / 2.
    bs = b * cam_bbox[:, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = (2 * (cx - w_2) / bs) + cam_bbox[:, 1]
    ty = (2 * (cy - h_2) / bs) + cam_bbox[:, 2]
    full_cam = torch.stack([tx, ty, tz], dim=-1)
    return full_cam

def get_light_poses(n_lights=5, elevation=np.pi / 3, dist=12):
    # get lights in a circle around origin at elevation
    thetas = elevation * np.ones(n_lights)
    phis = 2 * np.pi * np.arange(n_lights) / n_lights
    poses = []
    trans = make_translation(torch.tensor([0, 0, dist]))
    for phi, theta in zip(phis, thetas):
        rot = make_rotation(rx=-theta, ry=phi, order="xyz")
        poses.append((rot @ trans).numpy())
    return poses

def make_translation(t):
    return make_4x4_pose(torch.eye(3), t)

def make_rotation(rx=0, ry=0, rz=0, order="xyz"):
    Rx = rotx(rx)
    Ry = roty(ry)
    Rz = rotz(rz)
    if order == "xyz":
        R = Rz @ Ry @ Rx
    elif order == "xzy":
        R = Ry @ Rz @ Rx
    elif order == "yxz":
        R = Rz @ Rx @ Ry
    elif order == "yzx":
        R = Rx @ Rz @ Ry
    elif order == "zyx":
        R = Rx @ Ry @ Rz
    elif order == "zxy":
        R = Ry @ Rx @ Rz
    return make_4x4_pose(R, torch.zeros(3))

def make_4x4_pose(R, t):
    """
    :param R (*, 3, 3)
    :param t (*, 3)
    return (*, 4, 4)
    """
    dims = R.shape[:-2]
    pose_3x4 = torch.cat([R, t.view(*dims, 3, 1)], dim=-1)
    bottom = (
        torch.tensor([0, 0, 0, 1], device=R.device)
        .reshape(*(1,) * len(dims), 1, 4)
        .expand(*dims, 1, 4)
    )
    return torch.cat([pose_3x4, bottom], dim=-2)

def rotx(theta):
    return torch.tensor(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ],
        dtype=torch.float32,
    )

def roty(theta):
    return torch.tensor(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ],
        dtype=torch.float32,
    )

def rotz(theta):
    return torch.tensor(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )

def create_raymond_lights() -> List[pyrender.Node]:
    """
    Return raymond light nodes for the scene.
    """
    thetas = np.pi * np.array([1.0/6.0, 1.0/6.0, 1.0/6.0])
    phis = np.pi * np.array([0.0, 2.0/3.0, 4.0/3.0])
    nodes = []
    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)
        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, y, z]
        node = pyrender.Node(light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
                             matrix=matrix)
        nodes.append(node)
    return nodes

def alpha_composite(overlay_rgba: np.ndarray, main_rgba: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """
    단순 'cut-out' 방식 합성:
    - overlay의 alpha > threshold인 픽셀만 main 위를 덮어씀
    """
    final = main_rgba.copy()
    overlay_alpha = overlay_rgba[:, :, 3:4]
    mask = (overlay_alpha[..., 0] > threshold)
    final[mask, :3] = overlay_rgba[mask, :3]
    final[mask, 3]  = overlay_rgba[mask, 3]
    return final

class Renderer:
    def __init__(self, cfg: CfgNode, faces: np.array):
        """
        Wrapper around the pyrender renderer to render MANO meshes.
        Args:
            cfg (CfgNode): Model config file.
            faces (np.array): Array of shape (F, 3) containing the mesh faces.
        """
        self.cfg = cfg
        self.focal_length = cfg.EXTRA.FOCAL_LENGTH
        self.img_res = cfg.MODEL.IMAGE_SIZE

        # add faces that make the hand mesh watertight
        faces_new = np.array([
            [92, 38, 234],
            [234, 38, 239],
            [38, 122, 239],
            [239, 122, 279],
            [122, 118, 279],
            [279, 118, 215],
            [118, 117, 215],
            [215, 117, 214],
            [117, 119, 214],
            [214, 119, 121],
            [119, 120, 121],
            [121, 120, 78],
            [120, 108, 78],
            [78, 108, 79]
        ])
        faces = np.concatenate([faces, faces_new], axis=0)

        self.camera_center = [self.img_res // 2, self.img_res // 2]
        self.faces = faces
        # left hand의 경우 face 순서를 바꿔서 사용
        self.faces_left = self.faces[:, [0, 2, 1]]

    def __call__(self,
                 vertices: np.array,
                 camera_translation: np.array,
                 image: torch.Tensor,
                 full_frame: bool = False,
                 imgname: Optional[str] = None,
                 side_view=False, rot_angle=90,
                 mesh_base_color=(1.0, 1.0, 0.9),
                 scene_bg_color=(0, 0, 0),
                 return_rgba=False,
                 mesh_alpha=1.0,
                 # tactile 관련
                 tactile_values: Optional[List[float]] = None,
                 tactile_vertex_groups: Optional[dict] = None,
                 tactile_opacity: float = 0.7,
                 tactile_cmap: str = 'Reds',
                 is_right: int = 1,
                 # overlay 관련: 'joint', 'skin_point', 'skin_mesh'
                 overlay_type: str = 'skin_mesh',
                 joint_sphere_radius: float = 15.0,
                 joint_alpha: float = 1.0,
                 joint_cmap: str = 'Reds',
                 # 센서값 threshold (0~1)
                 tactile_sensor_threshold: float = 0.5,
                 # skeleton joint 기반 overlay 시, 21개 joint 좌표 (np.array, shape: (21,3))
                 joint_keypoints_3d: Optional[np.ndarray] = None
                 ) -> np.array:
        """
        2-패스 렌더링으로 단일 손을 시각화.
        1) 메인 손 메쉬만 렌더 -> main_rgba
        2) overlay (촉각 또는 joint/skin_point 방식)만 렌더 -> overlay_rgba
        3) alpha_composite -> final_rgba
        4) full_frame이 아니면 배경 이미지와 합성하여 반환
        """
        # --------------------
        # (1) 배경 이미지 준비
        # --------------------
        if full_frame:
            bg_img = cv2.imread(imgname).astype(np.float32)[:, :, ::-1] / 255.
        else:
            tmp = image.clone() * torch.tensor(self.cfg.MODEL.IMAGE_STD, device=image.device).reshape(3, 1, 1)
            tmp = tmp + torch.tensor(self.cfg.MODEL.IMAGE_MEAN, device=image.device).reshape(3, 1, 1)
            bg_img = tmp.permute(1, 2, 0).cpu().numpy()
        H, W = bg_img.shape[:2]

        camera_translation[0] *= -1.

        # --------------------
        # (2) 메인 손 메쉬 렌더링
        # --------------------
        scene_main = pyrender.Scene(bg_color=[*scene_bg_color, 0.0], ambient_light=(0.3, 0.3, 0.3))
        main_mesh = self._build_hand_mesh(vertices, is_right, side_view, rot_angle, mesh_base_color, mesh_alpha)
        scene_main.add(main_mesh)
        cam_pose, cam_node = self._add_camera_and_lights(scene_main, W, H, camera_translation)
        main_rgba = self._render_scene(scene_main, W, H)

        # --------------------
        # (3) overlay 렌더링 (overlay_type에 따라)
        # --------------------
        overlay_rgba = None
        if overlay_type == 'joint' and (tactile_values is not None) and (tactile_vertex_groups is not None):
            scene_overlay = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=(0, 0, 0))
            cam_pose_overlay, cam_node_overlay = self._add_camera_and_lights(scene_overlay, W, H, camera_translation)
            if joint_keypoints_3d is not None:
                joint_meshes = self._build_joint_spheres_from_skeleton(
                    vertices, is_right, side_view, rot_angle,
                    tactile_values, joint_keypoints_3d, tactile_vertex_groups,
                    joint_sphere_radius, joint_cmap, joint_alpha,
                    camera_translation, tactile_sensor_threshold
                )
            else:
                joint_meshes = self._build_joint_spheres(
                    vertices, is_right, side_view, rot_angle,
                    tactile_vertex_groups, tactile_values,
                    joint_sphere_radius, joint_cmap, joint_alpha,
                    camera_translation, tactile_sensor_threshold
                )
            for m in joint_meshes:
                scene_overlay.add(m)
            for node in create_raymond_lights():
                scene_overlay.add_node(node)
            overlay_rgba = self._render_scene(scene_overlay, W, H)
            final_rgba = alpha_composite(overlay_rgba, main_rgba)
        elif overlay_type == 'skin_point' and (tactile_values is not None) and (tactile_vertex_groups is not None):
            scene_overlay = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=(0, 0, 0))
            cam_pose_overlay, cam_node_overlay = self._add_camera_and_lights(scene_overlay, W, H, camera_translation)
            point_meshes = self._build_skin_points(
                vertices, is_right, side_view, rot_angle,
                tactile_values, tactile_vertex_groups,
                joint_sphere_radius, joint_cmap, joint_alpha,
                tactile_sensor_threshold
            )
            for m in point_meshes:
                scene_overlay.add(m)
            for node in create_raymond_lights():
                scene_overlay.add_node(node)
            overlay_rgba = self._render_scene(scene_overlay, W, H)
            final_rgba = alpha_composite(overlay_rgba, main_rgba)
        elif (tactile_values is not None) and (tactile_vertex_groups is not None):
            # skin_mesh mode (기존 face 기반 overlay)
            scene_overlay = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=(0.0, 0.0, 0.0))
            overlay_meshes = self._build_tactile_overlays(
                vertices, is_right, side_view, rot_angle,
                tactile_values, tactile_vertex_groups, tactile_opacity, tactile_cmap,
                tactile_sensor_threshold
            )
            for m in overlay_meshes:
                scene_overlay.add(m)
            cam_node_overlay = pyrender.Node(camera=cam_node.camera, matrix=cam_pose)
            scene_overlay.add_node(cam_node_overlay)
            for node in create_raymond_lights():
                scene_overlay.add_node(node)
            overlay_rgba = self._render_scene(scene_overlay, W, H)
            final_rgba = alpha_composite(overlay_rgba, main_rgba)
        else:
            final_rgba = main_rgba

        # --------------------
        # (4) 배경 합성 (side_view가 아니면)
        # --------------------
        if not side_view:
            valid_mask = final_rgba[:, :, 3:4]
            out_rgb = final_rgba[:, :, :3] * valid_mask + bg_img * (1 - valid_mask)
            out_rgb = out_rgb.astype(np.float32)
        else:
            out_rgb = final_rgba[:, :, :3]

        if return_rgba:
            return final_rgba
        return out_rgb

    def _build_hand_mesh(self, vertices, is_right, side_view, rot_angle, mesh_base_color, mesh_alpha):
        vtx = vertices.copy()
        if side_view:
            rot = trimesh.transformations.rotation_matrix(np.radians(rot_angle), [0, 1, 0])
            vtx = trimesh.transformations.transform_points(vtx, rot)
        rot180 = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        vtx = trimesh.transformations.transform_points(vtx, rot180)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='BLEND',
            baseColorFactor=(*mesh_base_color, mesh_alpha)
        )
        faces_to_use = self.faces if is_right else self.faces_left
        tri = trimesh.Trimesh(vtx, faces_to_use)
        mesh = pyrender.Mesh.from_trimesh(tri, material=material)
        return mesh

    def _build_tactile_overlays(self, vertices, is_right, side_view, rot_angle,
                                tactile_values, tactile_vertex_groups,
                                tactile_opacity, tactile_cmap,
                                tactile_sensor_threshold):
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt

        meshes = []
        norm = mcolors.Normalize(vmin=0, vmax=16)
        cmap_func = plt.get_cmap(tactile_cmap)

        vtx = vertices.copy()
        if side_view:
            rot_sv = trimesh.transformations.rotation_matrix(np.radians(rot_angle), [0, 1, 0])
            vtx = trimesh.transformations.transform_points(vtx, rot_sv)
        rot180 = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        vtx = trimesh.transformations.transform_points(vtx, rot180)
        faces_to_use = self.faces if is_right else self.faces_left

        for (group_name, indices), val in zip(tactile_vertex_groups.items(), tactile_values):
            # 센서값이 threshold 미만이면 건너뛰기
            if val < tactile_sensor_threshold:
                continue

            group_set = set(indices)
            group_faces = [f for f in faces_to_use if set(f).issubset(group_set)]
            if len(group_faces) == 0:
                continue

            rgba = cmap_func(norm(val))
            group_color = (*rgba[:3], tactile_opacity)

            sub_tri = trimesh.Trimesh(vertices=vtx.copy(), faces=np.array(group_faces))
            sub_mat = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.0,
                alphaMode='OPAQUE',
                baseColorFactor=group_color,
                doubleSided=True,
                emissiveFactor=(group_color[0], group_color[1], group_color[2])
            )
            sub_mesh = pyrender.Mesh.from_trimesh(sub_tri, material=sub_mat)
            meshes.append(sub_mesh)
        return meshes

    def _build_joint_spheres(self, vertices, is_right, side_view, rot_angle,
                              tactile_vertex_groups, sensor_values,
                              sphere_radius, joint_cmap, joint_alpha,
                              camera_translation,
                              tactile_sensor_threshold):
        """
        기존 tactile_vertex_groups 기반 joint overlay (tactile sensor 값에 따라 구의 크기를 조절)
        """
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt

        sphere_meshes = []
        norm = mcolors.Normalize(vmin=0, vmax=16)
        cmap_func = plt.get_cmap(joint_cmap)

        vtx = vertices.copy()
        if side_view:
            rot_sv = trimesh.transformations.rotation_matrix(np.radians(rot_angle), [0, 1, 0])
            vtx = trimesh.transformations.transform_points(vtx, rot_sv)
        rot180 = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        vtx = trimesh.transformations.transform_points(vtx, rot180)

        # 기준 거리: 카메라 위치와 원점 사이의 거리 (카메라가 [0,0,z]에 있다고 가정)
        ref_distance = np.linalg.norm(camera_translation) + 1e-6

        for (group_name, indices), sensor_val in zip(tactile_vertex_groups.items(), sensor_values):
            # 센서값이 threshold 미만이면 건너뛰기
            if sensor_val < tactile_sensor_threshold:
                continue

            group_indices = np.array(indices)
            if group_indices.max() >= vtx.shape[0]:
                continue
            joint_pos = vtx[group_indices].mean(axis=0)
            # 카메라와 joint 간 거리 계산
            distance = np.linalg.norm(joint_pos - camera_translation) + 1e-6
            # 원근법에 따른 스케일: 기준 거리일 때는 sphere_radius, 멀어지면 작게, 가까우면 크게
            scaled_radius = sphere_radius * (ref_distance / (distance)**2.1)
            rgba = cmap_func(norm(sensor_val))
            color = (*rgba[:3], joint_alpha)
            sphere = trimesh.creation.icosphere(subdivisions=2, radius=scaled_radius)
            sphere.apply_translation(joint_pos)
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.0,
                alphaMode='BLEND',
                baseColorFactor=color
            )
            sphere_mesh = pyrender.Mesh.from_trimesh(sphere, material=material)
            sphere_meshes.append(sphere_mesh)
        return sphere_meshes

    def _build_joint_spheres_from_skeleton(self, vertices, is_right, side_view, rot_angle,
                                    tactile_values, joint_keypoints_3d, tactile_vertex_groups,
                                    sphere_radius, joint_cmap, joint_alpha,
                                    camera_translation,
                                    tactile_sensor_threshold):
        """
        skeleton에서 추출한 21개 joint 좌표를 사용하여 구를 생성하고,
        tactile sensor 값이 24개인 경우, 나머지 3개는 joint_keypoints_3d를 이용해 palm 영역에서 추정한 위치로 표시.
        """
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        import numpy as np

        sphere_meshes = []
        norm = mcolors.Normalize(vmin=0, vmax=16)
        cmap_func = plt.get_cmap(joint_cmap)

        # joint_keypoints_3d에도 메쉬와 동일한 변환 적용
        transformed_joints = joint_keypoints_3d.copy()
        if side_view:
            rot_sv = trimesh.transformations.rotation_matrix(np.radians(rot_angle), [0, 1, 0])
            transformed_joints = trimesh.transformations.transform_points(transformed_joints, rot_sv)
        rot180 = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        transformed_joints = trimesh.transformations.transform_points(transformed_joints, rot180)

        # 첫 21개 센서: skeleton joint 좌표로 구 생성
        for i in range(21):
            sensor_val = tactile_values[i]
            if sensor_val < tactile_sensor_threshold:
                continue
            joint_pos = transformed_joints[i]
            ref_distance = np.linalg.norm(camera_translation) + 1e-6
            distance = np.linalg.norm(joint_pos - camera_translation) + 1e-6
            scaled_radius = sphere_radius * (ref_distance / (distance)**2.1)
            rgba = cmap_func(norm(sensor_val))
            color = (*rgba[:3], joint_alpha)
            sphere = trimesh.creation.icosphere(subdivisions=2, radius=scaled_radius)
            sphere.apply_translation(joint_pos)
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.0,
                alphaMode='BLEND',
                baseColorFactor=color
            )
            sphere_mesh = pyrender.Mesh.from_trimesh(sphere, material=material)
            sphere_meshes.append(sphere_mesh)

        # extra 3개 센서: palm 영역의 정보를 활용하여 위치 추정
        # palm 영역 대표 keypoint 인덱스 (예시: wrist, thumb_base, index_base, middle_base, ring_base, little_base)
        palm_indices = [0, 1, 5, 9, 13, 17]
        palm_center = transformed_joints[palm_indices].mean(axis=0)

        # 손의 palm normal 및 little finger 쪽 방향 계산
        wrist = transformed_joints[0]
        index_base = transformed_joints[9]  # 대표 index finger base
        little_base = transformed_joints[17]  # 대표 little finger base
        # palm 평면의 법선: wrist에서 index와 little finger 방향으로의 외적
        palm_normal = np.cross(index_base - wrist, little_base - wrist)
        palm_normal = palm_normal / (np.linalg.norm(palm_normal) + 1e-6)
        # little finger 쪽 방향: palm_center에서 little_base 방향
        side_dir = little_base - palm_center
        side_dir = side_dir / (np.linalg.norm(side_dir) + 1e-6)
        # 만약 왼손이면, 좌우 방향을 반전하여 little finger 쪽이 맞게 보정
        if not is_right:
            side_dir = -side_dir

        # 손 크기(손바닥 영역 keypoint들과의 거리)를 측정하여 offset 산출
        hand_dists = [np.linalg.norm(transformed_joints[i] - palm_center) for i in palm_indices]
        hand_scale = np.max(hand_dists)  # 또는 np.mean(hand_dists)
        offset = hand_scale * 0.3
        # offset 값을 손 크기의 비율로 결정 (예: 15%와 30% 정도)
        # 기존 extra 센서 위치 두 개는 그대로 유지:
        #   extra_positions[0]: A = transformed_joints[0] + side_dir * offset_high (Palm7 위치)
        #   extra_positions[1]: palm_center (중앙)
        A = transformed_joints[0] + side_dir * offset  if is_right else transformed_joints[0] - side_dir * offset
        # little finger의 손바닥 joint: 여기서는 transformed_joints[17]로 가정 (Palm의 little finger 쪽)
        B = transformed_joints[17]
        # palm_center를 P라 할 때, P의 A-B 선 위 직교 투영을 계산
        P = palm_center
        t = np.dot(P - A, B - A) / np.dot(B - A, B - A)
        t = np.clip(t, 0, 1)  # 선분 내에 있도록 클램핑 (필요에 따라)
        extra_pos_projection = A + t * (B - A)

        extra_positions = [
            A,                   # 그대로: Palm7
            palm_center,         # 그대로: 중앙
            extra_pos_projection # 새로 계산된 위치 (palm_center의 투영)
        ]

        for j, extra_pos in enumerate(extra_positions):
            sensor_index = 21 + j
            sensor_val = tactile_values[sensor_index]
            if sensor_val < tactile_sensor_threshold:
                continue
            ref_distance = np.linalg.norm(camera_translation) + 1e-6
            distance = np.linalg.norm(extra_pos - camera_translation) + 1e-6
            scaled_radius = sphere_radius * (ref_distance / (distance)**2.1)
            rgba = cmap_func(norm(sensor_val))
            color = (*rgba[:3], joint_alpha)
            sphere = trimesh.creation.icosphere(subdivisions=2, radius=scaled_radius)
            sphere.apply_translation(extra_pos)
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.0,
                alphaMode='BLEND',
                baseColorFactor=color
            )
            sphere_mesh = pyrender.Mesh.from_trimesh(sphere, material=material)
            sphere_meshes.append(sphere_mesh)

        return sphere_meshes



    def _build_skin_points(self, vertices, is_right, side_view, rot_angle,
                           tactile_values, tactile_vertex_groups,
                           point_radius, joint_cmap, joint_alpha,
                           tactile_sensor_threshold):
        """
        tactile_vertex_groups 기반 각 그룹의 평균 위치에 고정 크기(point_radius)의 구체(포인트)를 생성.
        """
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        point_meshes = []
        norm = mcolors.Normalize(vmin=0, vmax=16)
        cmap_func = plt.get_cmap(joint_cmap)

        vtx = vertices.copy()
        if side_view:
            rot_sv = trimesh.transformations.rotation_matrix(np.radians(rot_angle), [0, 1, 0])
            vtx = trimesh.transformations.transform_points(vtx, rot_sv)
        rot180 = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        vtx = trimesh.transformations.transform_points(vtx, rot180)

        for (group_name, indices), sensor_val in zip(tactile_vertex_groups.items(), tactile_values):
            if sensor_val < tactile_sensor_threshold:
                continue
            group_vtx = vtx[indices]
            center_pos = group_vtx.mean(axis=0)
            rgba = cmap_func(norm(sensor_val))
            color = (*rgba[:3], joint_alpha)
            # 고정된 point_radius 사용
            sphere = trimesh.creation.icosphere(subdivisions=2, radius=point_radius)
            sphere.apply_translation(center_pos)
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.0,
                alphaMode='BLEND',
                baseColorFactor=color
            )
            sphere_mesh = pyrender.Mesh.from_trimesh(sphere, material=material)
            point_meshes.append(sphere_mesh)
        return point_meshes

    def _add_camera_and_lights(self, scene, width, height, camera_translation):
        cam_pose = np.eye(4)
        cam_trans = camera_translation.copy()
        cam_pose[:3, 3] = cam_trans
        camera_center = [width / 2., height / 2.]
        cam = pyrender.IntrinsicsCamera(
            fx=self.focal_length, fy=self.focal_length,
            cx=camera_center[0], cy=camera_center[1],
            zfar=1e12
        )
        cam_node = pyrender.Node(camera=cam, matrix=cam_pose)
        scene.add_node(cam_node)
        for node in create_raymond_lights():
            scene.add_node(node)
        return cam_pose, cam_node

    def _render_scene(self, scene, width, height):
        renderer = pyrender.OffscreenRenderer(viewport_width=width,
                                              viewport_height=height,
                                              point_size=1.0)
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        renderer.delete()
        return color.astype(np.float32) / 255.0

    def vertices_to_trimesh(self,
                            vertices, camera_translation,
                            mesh_base_color=(1.0, 1.0, 0.9),
                            rot_axis=[1, 0, 0], rot_angle=0,
                            is_right=1, mesh_alpha=1.0):
        vertex_colors = np.array([(*mesh_base_color, mesh_alpha)] * vertices.shape[0])
        vtx = vertices.copy()
        if is_right == 0:
            vtx[:, 0] = -vtx[:, 0]
        vtx = vtx + camera_translation
        rot = trimesh.transformations.rotation_matrix(np.radians(rot_angle), rot_axis)
        vtx = trimesh.transformations.transform_points(vtx, rot)
        rot180 = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        vtx = trimesh.transformations.transform_points(vtx, rot180)
        faces_to_use = self.faces if is_right else self.faces_left
        mesh = trimesh.Trimesh(vtx, faces_to_use, vertex_colors=vertex_colors)
        return mesh

    def render_rgba(self,
                    vertices: np.array,
                    cam_t=None,
                    rot=None,
                    rot_axis=[1, 0, 0],
                    rot_angle=0,
                    camera_z=3,
                    mesh_base_color=(1.0, 1.0, 0.9),
                    scene_bg_color=(0, 0, 0),
                    render_res=[256, 256],
                    focal_length=None,
                    is_right=None):
        renderer = pyrender.OffscreenRenderer(viewport_width=render_res[0],
                                              viewport_height=render_res[1],
                                              point_size=1.0)
        focal_length = focal_length if focal_length is not None else self.focal_length
        if cam_t is not None:
            camera_translation = cam_t.copy()
        else:
            camera_translation = np.array([0, 0, camera_z * focal_length / render_res[1]])
        tri_mesh = self.vertices_to_trimesh(vertices, np.array([0, 0, 0]),
                                            mesh_base_color, rot_axis, rot_angle,
                                            is_right=is_right, mesh_alpha=1.0)
        mesh = pyrender.Mesh.from_trimesh(tri_mesh)
        scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh)
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera_center = [render_res[0] / 2., render_res[1] / 2.]
        camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length,
                                           cx=camera_center[0], cy=camera_center[1], zfar=1e12)
        cam_node = pyrender.Node(camera=camera, matrix=camera_pose)
        scene.add_node(cam_node)
        for node in create_raymond_lights():
            scene.add_node(node)
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        renderer.delete()
        return color.astype(np.float32) / 255.0

    def render_rgba_multiple(self,
                            vertices: List[np.array],
                            cam_t: List[np.array],
                            rot_axis=[1, 0, 0],
                            rot_angle=0,
                            mesh_base_color=(1.0, 1.0, 0.9),
                            scene_bg_color=(0, 0, 0),
                            render_res=[256, 256],
                            focal_length=None,
                            is_right=None,
                            # 기존 tactile overlay 관련
                            tactile_values_list: Optional[List[List[float]]] = None,
                            tactile_vertex_groups: Optional[dict] = None,
                            tactile_cmap: str = 'viridis',
                            tactile_opacity: float = 0.7,
                            mesh_alpha: float = 0.3,
                            # 새 overlay 관련
                            overlay_type: str = 'skin_mesh',
                            joint_sphere_radius: float = 5.0,
                            joint_alpha: float = 1.0,
                            joint_cmap: str = 'viridis',
                            # skeleton joint overlay 시, 각 사람별 joint 좌표 (List[np.array], 각 np.array shape: (21,3))
                            joint_keypoints_3d_list: Optional[List[np.array]] = None,
                            # 센서값 threshold (0~1)
                            tactile_sensor_threshold: float = 0.5):
        """
        여러 손(people)을 동시에 렌더링. overlay_type에 따라
        - 'joint':   skeleton joint 기반 overlay (추가 sensor 값은 tactile_vertex_groups의 Palm7,8,9 사용)
        - 'skin_point': tactile sensor 그룹의 평균 위치에 point marker 표시
        - 'skin_mesh': 기존 face-based overlay 방식
        """
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt

        renderer = pyrender.OffscreenRenderer(viewport_width=render_res[0],
                                              viewport_height=render_res[1],
                                              point_size=1.0)

        if is_right is None:
            is_right = [1 for _ in range(len(vertices))]

        # -------------------------------------------------------
        # (1) 메인 손 메쉬를 모두 한 장면(scene_main)에 렌더링
        # -------------------------------------------------------
        scene_main = pyrender.Scene(bg_color=[*scene_bg_color, 0.0],
                                    ambient_light=(0.3, 0.3, 0.3))

        for i, (vvv, ttt, sss) in enumerate(zip(vertices, cam_t, is_right)):
            tri = self.vertices_to_trimesh(
                vertices=vvv,
                camera_translation=ttt.copy(),
                mesh_base_color=mesh_base_color,
                rot_axis=rot_axis,
                rot_angle=rot_angle,
                is_right=sss,
                mesh_alpha=mesh_alpha
            )
            main_mesh = pyrender.Mesh.from_trimesh(tri)
            scene_main.add(main_mesh)

        camera_pose = np.eye(4)
        camera_center = [render_res[0] / 2., render_res[1] / 2.]
        focal_length_final = focal_length if focal_length is not None else self.focal_length
        camera = pyrender.IntrinsicsCamera(fx=focal_length_final, fy=focal_length_final,
                                           cx=camera_center[0], cy=camera_center[1], zfar=1e12)
        cam_node = pyrender.Node(camera=camera, matrix=camera_pose)
        scene_main.add_node(cam_node)

        # 기본 조명 추가
        for node in create_raymond_lights():
            scene_main.add_node(node)

        # 메인 렌더링
        main_color, _ = renderer.render(scene_main, flags=pyrender.RenderFlags.RGBA)
        main_color = main_color.astype(np.float32) / 255.0

        # -------------------------------------------------------
        # (2) overlay (joint 또는 tactile)를 별도 scene에 렌더링
        # -------------------------------------------------------
        if (tactile_values_list is None) or (tactile_vertex_groups is None):
            renderer.delete()
            return main_color  # 오버레이 없음

        scene_overlay = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=(0, 0, 0))
        cam_node_overlay = pyrender.Node(camera=camera, matrix=camera_pose)
        scene_overlay.add_node(cam_node_overlay)
        for node in create_raymond_lights():
            scene_overlay.add_node(node)

        if overlay_type == 'joint':
            norm = mcolors.Normalize(vmin=0, vmax=10)
            cmap_func = plt.get_cmap(joint_cmap)

            for i, (vvv, ttt, sss) in enumerate(zip(vertices, cam_t, is_right)):
                sensor_vals = tactile_values_list[i]
                # vertices 변환: 왼손 flip, translation, 회전, 등
                vtx = vvv.copy()
                if sss == 0:
                    vtx[:, 0] = -vtx[:, 0]
                vtx = vtx + ttt.copy()
                rot = trimesh.transformations.rotation_matrix(np.radians(rot_angle), rot_axis)
                vtx = trimesh.transformations.transform_points(vtx, rot)
                rot180 = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
                vtx = trimesh.transformations.transform_points(vtx, rot180)

                ref_distance = np.linalg.norm(ttt) + 1e-6

                # skeleton joint overlay: joint_keypoints_3d_list가 제공되면 사용, 아니면 tactile_vertex_groups 기반 사용
                if (joint_keypoints_3d_list is not None) and (joint_keypoints_3d_list[i] is not None):
                    # skeleton joint 좌표에 flip, translation, 회전 적용 (이미 위에서 vtx에 대해 처리한 rot, rot180과 동일)
                    joint_keypoints = joint_keypoints_3d_list[i].copy()
                    if sss == 0:
                        joint_keypoints[:, 0] = -joint_keypoints[:, 0]
                    joint_keypoints = joint_keypoints + ttt.copy()
                    joint_keypoints = trimesh.transformations.transform_points(joint_keypoints, rot)
                    transformed_joints = trimesh.transformations.transform_points(joint_keypoints, rot180)

                    # 첫 21개: skeleton joint 좌표로 구 생성
                    for j in range(21):
                        if sensor_vals[j] < tactile_sensor_threshold:
                            continue
                        joint_pos = transformed_joints[j]
                        distance = np.linalg.norm(joint_pos) + 1e-6
                        scaled_radius = joint_sphere_radius * (ref_distance / (distance)**2.1)
                        rgba = cmap_func(norm(sensor_vals[j]))
                        color = (*rgba[:3], joint_alpha)
                        sphere = trimesh.creation.icosphere(subdivisions=2, radius=scaled_radius)
                        sphere.apply_translation(joint_pos)
                        mat = pyrender.MetallicRoughnessMaterial(
                            metallicFactor=0.0,
                            alphaMode='OPAQUE',
                            baseColorFactor=color,
                        )
                        mesh_sphere = pyrender.Mesh.from_trimesh(sphere, material=mat)
                        scene_overlay.add(mesh_sphere)
                    
                    # extra 3개 센서: palm 영역의 정보를 활용하여 위치 추정
                    # palm 영역 대표 keypoint 인덱스 (예시: wrist, thumb_base, index_base, middle_base, ring_base, little_base)
                    palm_indices = [0, 1, 5, 9, 13, 17]
                    palm_center = transformed_joints[palm_indices].mean(axis=0)

                    # 손의 palm normal 및 little finger 쪽 방향 계산
                    wrist = transformed_joints[0]
                    index_base = transformed_joints[9]  # 대표 index finger base
                    little_base = transformed_joints[17]  # 대표 little finger base
                    # palm 평면의 법선: wrist에서 index와 little finger 방향으로의 외적
                    palm_normal = np.cross(index_base - wrist, little_base - wrist)
                    palm_normal = palm_normal / (np.linalg.norm(palm_normal) + 1e-6)
                    # little finger 쪽 방향: palm_center에서 little_base 방향
                    side_dir = little_base - palm_center
                    side_dir = side_dir / (np.linalg.norm(side_dir) + 1e-6)
                    # 만약 왼손이면, 좌우 방향을 반전하여 little finger 쪽이 맞게 보정
                    if sss == 0:
                        side_dir = -side_dir

                    # 손 크기(손바닥 영역 keypoint들과의 거리)를 측정하여 offset 산출
                    hand_dists = [np.linalg.norm(transformed_joints[i] - palm_center) for i in palm_indices]
                    hand_scale = np.max(hand_dists)  # 또는 np.mean(hand_dists)
                    offset = hand_scale * 0.3
                    # offset 값을 손 크기의 비율로 결정 (예: 15%와 30% 정도)
                    # 기존 extra 센서 위치 두 개는 그대로 유지:
                    #   extra_positions[0]: A = transformed_joints[0] + side_dir * offset_high (Palm7 위치)
                    #   extra_positions[1]: palm_center (중앙)
                    A = transformed_joints[0] + side_dir * offset  if is_right else transformed_joints[0] - side_dir * offset
                    # little finger의 손바닥 joint: 여기서는 transformed_joints[17]로 가정 (Palm의 little finger 쪽)
                    B = transformed_joints[17]
                    # palm_center를 P라 할 때, P의 A-B 선 위 직교 투영을 계산
                    P = palm_center
                    t = np.dot(P - A, B - A) / np.dot(B - A, B - A)
                    t = np.clip(t, 0, 1)  # 선분 내에 있도록 클램핑 (필요에 따라)
                    extra_pos_projection = A + t * (B - A)

                    extra_positions = [
                        A,                   # 그대로: Palm7
                        palm_center,         # 그대로: 중앙
                        extra_pos_projection # 새로 계산된 위치 (palm_center의 투영)
                    ]
                    for sensor_index, extra_pos in zip(range(21, 24), extra_positions):
                        if sensor_vals[sensor_index] < tactile_sensor_threshold:
                            continue
                        distance = np.linalg.norm(extra_pos) + 1e-6
                        scaled_radius = joint_sphere_radius * (ref_distance / (distance)**2.1)
                        rgba = cmap_func(norm(sensor_vals[sensor_index]))
                        color = (*rgba[:3], joint_alpha)
                        sphere = trimesh.creation.icosphere(subdivisions=2, radius=scaled_radius)
                        sphere.apply_translation(extra_pos)
                        mat = pyrender.MetallicRoughnessMaterial(
                            metallicFactor=0.0,
                            alphaMode='BLEND',
                            baseColorFactor=color,
                        )
                        mesh_sphere = pyrender.Mesh.from_trimesh(sphere, material=mat)
                        scene_overlay.add(mesh_sphere)

                else:
                    # tactile_vertex_groups 기반 joint overlay (기존 방식)
                    for group_idx, (group_name, indices) in enumerate(tactile_vertex_groups.items()):
                        if sensor_vals[group_idx] < tactile_sensor_threshold:
                            continue
                        rgba = cmap_func(norm(sensor_vals[group_idx]))
                        color = (*rgba[:3], joint_alpha)
                        group_vtx = vtx[indices]
                        center_pos = group_vtx.mean(axis=0)
                        distance = np.linalg.norm(center_pos) + 1e-6
                        scaled_radius = joint_sphere_radius * (ref_distance / (distance)**2.1)
                        sphere = trimesh.creation.icosphere(subdivisions=2, radius=scaled_radius)
                        sphere.apply_translation(center_pos)
                        mat = pyrender.MetallicRoughnessMaterial(
                            metallicFactor=0.0,
                            alphaMode='BLEND',
                            baseColorFactor=color,
                        )
                        mesh_sphere = pyrender.Mesh.from_trimesh(sphere, material=mat)
                        scene_overlay.add(mesh_sphere)

        elif overlay_type == 'skin_point':
            norm = mcolors.Normalize(vmin=0, vmax=16)
            cmap_func = plt.get_cmap(joint_cmap)
            for i, (vvv, ttt, sss) in enumerate(zip(vertices, cam_t, is_right)):
                sensor_vals = tactile_values_list[i]
                vtx = vvv.copy()
                if sss == 0:
                    vtx[:, 0] = -vtx[:, 0]
                vtx = vtx + ttt.copy()
                rot = trimesh.transformations.rotation_matrix(np.radians(rot_angle), rot_axis)
                vtx = trimesh.transformations.transform_points(vtx, rot)
                rot180 = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
                vtx = trimesh.transformations.transform_points(vtx, rot180)

                ref_distance = np.linalg.norm(ttt) + 1e-6

                for group_idx, (group_name, indices) in enumerate(tactile_vertex_groups.items()):
                    if sensor_vals[group_idx] < tactile_sensor_threshold:
                        continue
                    rgba = cmap_func(norm(sensor_vals[group_idx]))
                    color = (*rgba[:3], joint_alpha)
                    group_vtx = vtx[indices]
                    center_pos = group_vtx.mean(axis=0)
                    distance = np.linalg.norm(center_pos) + 1e-6
                    scaled_radius = joint_sphere_radius * (ref_distance / (distance)**2.1)
                    sphere = trimesh.creation.icosphere(subdivisions=2, radius=scaled_radius)                    # 고정된 반지름 사용
                    sphere.apply_translation(center_pos)
                    mat = pyrender.MetallicRoughnessMaterial(
                        metallicFactor=0.0,
                        alphaMode='BLEND',
                        baseColorFactor=color,
                    )
                    mesh_sphere = pyrender.Mesh.from_trimesh(sphere, material=mat)
                    scene_overlay.add(mesh_sphere)
        else:  # skin_mesh
            norm = mcolors.Normalize(vmin=0, vmax=16)
            cmap_func = plt.get_cmap(tactile_cmap)
            for i, (vvv, ttt, sss) in enumerate(zip(vertices, cam_t, is_right)):
                sensor_vals = tactile_values_list[i]
                vtx = vvv.copy()
                if sss == 0:
                    vtx[:, 0] = -vtx[:, 0]
                vtx = vtx + ttt.copy()
                rot = trimesh.transformations.rotation_matrix(np.radians(rot_angle), rot_axis)
                vtx = trimesh.transformations.transform_points(vtx, rot)
                rot180 = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
                vtx = trimesh.transformations.transform_points(vtx, rot180)

                faces_to_use = self.faces if sss else self.faces_left

                for group_idx, (group_name, indices) in enumerate(tactile_vertex_groups.items()):
                    if sensor_vals[group_idx] < tactile_sensor_threshold:
                        continue

                    group_set = set(indices)
                    group_faces = [face for face in faces_to_use if set(face).issubset(group_set)]
                    if len(group_faces) == 0:
                        continue

                    val = sensor_vals[group_idx]
                    rgba = cmap_func(norm(val))
                    group_color = (*rgba[:3], tactile_opacity)

                    sub_tri = trimesh.Trimesh(vertices=vtx, faces=np.array(group_faces))
                    sub_mat = pyrender.MetallicRoughnessMaterial(
                        metallicFactor=0.0,
                        alphaMode='OPAQUE',
                        baseColorFactor=group_color,
                        doubleSided=True,
                        emissiveFactor=(group_color[0], group_color[1], group_color[2])
                    )
                    sub_mesh = pyrender.Mesh.from_trimesh(sub_tri, material=sub_mat)
                    scene_overlay.add(sub_mesh)
        # overlay 렌더링
        overlay_color, _ = renderer.render(scene_overlay, flags=pyrender.RenderFlags.RGBA)
        overlay_color = overlay_color.astype(np.float32) / 255.0

        # -------------------------------------------------------
        # (3) 알파 합성
        # -------------------------------------------------------
        final_rgba = alpha_composite(overlay_color, main_color, threshold=0.1)

        renderer.delete()
        return final_rgba

    def add_lighting(self, scene, cam_node, color=np.ones(3), intensity=1.0):
        light_poses = get_light_poses()
        light_poses.append(np.eye(4))
        cam_pose = scene.get_pose(cam_node)
        for i, pose in enumerate(light_poses):
            matrix = cam_pose @ pose
            node = pyrender.Node(
                light=pyrender.DirectionalLight(color=color, intensity=intensity),
                matrix=matrix
            )
            scene.add_node(node)

    def add_point_lighting(self, scene, cam_node, color=np.ones(3), intensity=1.0):
        light_poses = get_light_poses(dist=0.5)
        light_poses.append(np.eye(4))
        cam_pose = scene.get_pose(cam_node)
        for i, pose in enumerate(light_poses):
            matrix = cam_pose @ pose
            node = pyrender.Node(
                light=pyrender.PointLight(color=color, intensity=intensity),
                matrix=matrix
            )
            scene.add_node(node)
