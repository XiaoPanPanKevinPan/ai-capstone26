import os
import re
import glob
import numpy as np
import open3d as o3d
import argparse
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
import time

# ---------- Camera Intrinsics (Resolution 512x512, FOV 90) ----------
# These parameters are derived from the Habitat pinhole camera model [cite: 26-27].
IMG_W, IMG_H = 512, 512
FOV = np.deg2rad(90.0)
FX = (IMG_W / 2.0) / np.tan(FOV / 2.0)
FY = (IMG_H / 2.0) / np.tan(FOV / 2.0)
CX, CY = IMG_W / 2.0, IMG_H / 2.0
DEPTH_SCALE = 1 / 10 * 255 # the same as load.py

# I think it should be renamed as RGBD_to_point_cloud, but I'll just keep it
def depth_image_to_point_cloud(rgb_image: str, depth_image: str):
    """
    TASK 1: Geometric Unprojection [cite: 12, 25-27]
    Convert depth pixels (u, v, d) into 3D world points (x, y, z).
    """
    # 1. Read & convert inputs to numpy arrays
    rgb_image = np.array(o3d.io.read_image(rgb_image))
    depth_image = np.array(o3d.io.read_image(depth_image))

    # 2. Convert depth to meters (Habitat depth is often scaled or normalized)
    depth_image = depth_image / DEPTH_SCALE

    # 3. Create a coordinate grid for (u, v) pixels
    u, v = np.meshgrid(np.arange(IMG_W), np.arange(IMG_H))
    
    # TODO: Implement unprojection logic here
    # 4. unprojection
    x = (u - CX) * depth_image / FX
    y = -(v - CY) * depth_image / FY # for v, up is [-]; for y, up is [+]
    z = -depth_image # (assuming camera looks towards -Z)

    # 5. before point cloud creation
    valid_mask = depth_image.flatten() > 0
    colors_norm = (rgb_image / 255.0).reshape(-1, 3)[valid_mask]
    points_3d = np.stack([
        x.flatten()[valid_mask], 
        y.flatten()[valid_mask], 
        z.flatten()[valid_mask]
    ], axis=1)

    # 6. create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors_norm)
    return pcd

def preprocess_point_cloud(pcd, voxel_size):
    """
    Pre-processing: Voxelization and Normal Estimation [cite: 17, 29]
    """

    # 1. Downsampling to
    # - reduce computational cost
    # - remove noise (especially for depth)
    # - make the point cloud more uniform
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    # TODO: Estimate normals for pcd_down (required for Point-to-Plane ICP)

    # 2. Estimate normals
    # - Note: Pt-to-Pl ICP is faster than Pt-to-Pt ICP
    #    ref: https://learnopencv.com/iterative-closest-point-icp-explained/
    radius_estNormal = voxel_size * 2.5
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_estNormal, max_nn=30)
    )
    
    # Compute FPFH features for Global Registration [cite: 30]
    radius_feature = voxel_size * 5.0
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return pcd_down, pcd_fpfh

def my_local_icp_algorithm(source_pcd, target_pcd, initial_transform):
    """
    TASK 2: Custom ICP Implementation (BONUS 20%) 
    Implement your own version of Point-to-Plane ICP.
    """
    T_global = initial_transform.copy()
    
    # TODO: Implement the ICP loop:
    # 1. Find nearest neighbors using target_tree.search_knn_vector_3d
    # 2. Build the linear system (AtA)x = Atb
    # 3. Solve for pose update and update T_global
    
    result = o3d.pipelines.registration.RegistrationResult()
    result.transformation = T_global
    return result

def local_icp_algorithm(source_down, target_down, trans_init, threshold):
    """
    TASK 2: Open3D ICP Implementation (REQUIRED) [cite: 32]
    """
    # TODO: Use o3d.pipelines.registration.registration_icp
    # Estimation method should be TransformationEstimationPointToPlane()

    # trans_init is the initial guess preventing ICP from getting stuck in local minima
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_down, target_down, threshold, trans_init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
    )
    
    return reg_p2p

def visualize_and_evaluate(reconstructed_pcd, predicted_cam_poses, gt_poses, args):
    """
    TASK 3: Evaluation & Visualization [cite: 19, 35-38]
    """
    # 1. Create LineSet for estimated trajectory (Red)
    pred_lines = o3d.geometry.LineSet()

    # Note: We'll use Vec3dVec (3 := the vec is (n, 3). d := the vec is float64)
    #       and Vec2iVec (2 := (n, 2). i := int32) to wrap the data

    # 1.1. Extract the points from predicted_cam_poses
    pred_lines.points = o3d.utility.Vector3dVector(
        predicted_cam_poses[:, 0:3, 3] 
        # each in predicted_cam_poses is a 4x4 homogeneous transformation matrix
        #   $^{world} T _{cam's local coord}$
        # here, 
        #   for each cam pose([:]), 
        #   extract [x, y, z] axis' ([row=0:3])
        #   translation part ([col=3])
    ) 

    # 1.2. Connect each pair of points with their indeces
    pred_lines.lines = o3d.utility.Vector2iVector(np.array(
        [ [i, i+1] for i in range(len(predicted_cam_poses)-1) ]
    ))

    # 1.3. Set the color of each line
    pred_lines.colors = o3d.utility.Vector3dVector(
        [ [1, 0, 0] for _ in range(len(predicted_cam_poses)-1) ]
    )
    
    # 2. Create LineSet for ground truth trajectory (Black)
    gt_lines = o3d.geometry.LineSet()

    # 2.1. Extract the points from gt_poses
    gt_lines.points = o3d.utility.Vector3dVector(
        gt_poses[:, 0:3, 3]
    )

    # 2.2. Connect each pair of points with their indeces
    gt_lines.lines = o3d.utility.Vector2iVector(np.array(
        [ [i, i+1] for i in range(len(gt_poses)-1) ]
    ))

    # 2.3. Set the color of each line
    gt_lines.colors = o3d.utility.Vector3dVector(
        [ [0, 0, 0] for _ in range(len(gt_poses)-1) ]
    )
    
    # TODO: Calculate Mean L2 Distance between predicted_cam_poses and gt_poses [cite: 38]
    # L2 = sqrt(dx^2 + dy^2 + dz^2)
    mean_l2_error = np.mean(
        np.sqrt(
            np.sum(
                (predicted_cam_poses[:, 0:3, 3] - gt_poses[:, 0:3, 3]) ** 2, axis=1
            )
        )
    )
    
    print(f"Mean L2 distance: {mean_l2_error:.6f} meters")
    
    # 3. Visualization
    o3d.visualization.draw_geometries([reconstructed_pcd, pred_lines, gt_lines], 
                                      window_name=f"Floor {args.floor} Reconstruction")
    return mean_l2_error

def cut_pcd(pcd, center, radius=2):
    min_bound = center - np.array([radius] * 3)
    max_bound = center + np.array([radius] * 3)
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    return pcd.crop(bbox)

def reconstruct(args):
    voxel_size = 0.20                           # for threshold and display
    voxel_size_in_progress = voxel_size / 2     # prevent loosing too much info
    rgb_dir = os.path.join(args.data_root, "rgb")
    depth_dir = os.path.join(args.data_root, "depth")

    rgb_files = sorted(
        glob.glob(os.path.join(rgb_dir, "*.png")), 
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )
    depth_files = sorted(
        glob.glob(os.path.join(depth_dir, "*.png")), 
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )
    
    # Load Ground Truth Poses [cite: 24, 54]
    gt_pose_path = os.path.join(args.data_root, "GT_pose.npy")
    gt_poses = []
    if os.path.exists(gt_pose_path):
        gt_data = np.load(gt_pose_path)
        for p in gt_data:
            mat = np.eye(4)
            mat[:3, :3] = R.from_quat([p[4], p[5], p[6], p[3]]).as_matrix()
            mat[:3, 3] = [p[0], p[1], p[2]]
            gt_poses.append(mat)
        gt_poses = np.stack(gt_poses)

    camera_poses = [np.eye(4)]
    accumulated_pcd = o3d.geometry.PointCloud()

    # Reconstruction Loop [cite: 29-30]
    target_pcd, target_fpfh = preprocess_point_cloud(
        depth_image_to_point_cloud(rgb_files[0], depth_files[0]), 
        voxel_size_in_progress
    )
    accumulated_pcd += target_pcd

    for i in range(1, len(rgb_files)):
        print(f"Processing Frame {i}...")
        # TODO: Implement the full pipeline:
        # 1. Convert RGB-D to PointCloud (Task 1)
        pcd = depth_image_to_point_cloud(rgb_files[i], depth_files[i])

        # 2. Preprocess (Voxel/FPFH/Normals)
        pcd, pcd_fpfh = preprocess_point_cloud(pcd, voxel_size_in_progress)

        # 3. Execute Global Registration (RANSAC)
        # step_movement_distance_bound = 1.50 
        #     # 0.25 is the default move_forward distance
        #     # while 10 is the default turn_left/turn_right angle in degrees
        # ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        #     source=pcd_down, target=prev_pcd_down, 
        #     source_feature=pcd_fpfh, target_feature=prev_pcd_fpfh, 
        #     mutual_filter=False,
        #     max_correspondence_distance=step_movement_distance_bound,
        #     # estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        #     # ransac_n=3, 
        #     checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9)],
        #     # criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=100000, confidence=0.999)
        # )

        # 4. Execute Local Registration (ICP - Task 2)
        # local_icp_result = local_icp_algorithm(
        #     pcd_down, prev_pcd_down, 
        #     ransac_result.transformation, 
        #     threshold=voxel_size_for_display
        # )
        local_icp_result = local_icp_algorithm(
            pcd, target_pcd, 
            camera_poses[-1], 
            threshold=voxel_size
        )
        
        # 5. Update camera_poses
        # camera_pose = camera_poses[-1] @ local_icp_result.transformation
        camera_pose = local_icp_result.transformation
        camera_poses.append(camera_pose)

        # 6. prepare for next iteration
        pcd_in_global = deepcopy(pcd)
        pcd_in_global.transform(camera_pose)

        target_pcd = cut_pcd(accumulated_pcd, camera_pose[:3, 3]) + pcd_in_global
        target_pcd, target_fpfh = preprocess_point_cloud(target_pcd, voxel_size_in_progress)

        # 7. accum pcd and optionally downsample accum to prevent OOM
        accumulated_pcd += pcd_in_global
        if i % 50 == 0:
            accumulated_pcd = accumulated_pcd.voxel_down_sample(voxel_size_in_progress)

    camera_poses = np.stack(camera_poses)
    accumulated_pcd = accumulated_pcd.voxel_down_sample(voxel_size)

    # TODO: Post-processing: remove the ceiling [cite: 37]
    # remove the ceiling by cutting the y axis for y > 0.8
    # bbox = accumulated_pcd.get_axis_aligned_bounding_box()
    # max_bound = bbox.get_max_bound()
    # max_bound[1] = 0.8  # Limit the maximum Y (height) to 0.8
    # bbox.max_bound = max_bound
    # accumulated_pcd = accumulated_pcd.crop(bbox)
    
    return accumulated_pcd, camera_poses, gt_poses

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--floor', type=int, default=1)
    parser.add_argument('-v', '--version', type=str, default='open3d', help='open3d or my_icp')
    args = parser.parse_args()

    # Set data root based on floor
    args.data_root = f"data_collection/first_floor/" if args.floor == 1 else f"data_collection/second_floor/"

    start_time = time.time()
    result_pcd, pred_poses, gt_poses = reconstruct(args)
    
    print(f"Total execution time: {time.time() - start_time:.2f}s") # 
    visualize_and_evaluate(result_pcd, pred_poses, gt_poses, args)