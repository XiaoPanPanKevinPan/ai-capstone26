import cv2
import numpy as np
from typing import List, Tuple

SCALE_FACTOR = 10000.0 / 255.0
# CEILING_COLOR = np.array([8, 255, 214])
# FLOOR_COLOR = np.array([255, 194, 7])
MAP_RESOLUTION = 20 # resolution: 5cm per pixel


def load_and_filter_map(point_path: str, color_path: str):

    points = np.load(point_path)
    colors = np.load(color_path)

    # Convert to real-world meters
    coords = points * SCALE_FACTOR

    # =============== TODO 1-1 ===============
    # Hints: To get a good 2d map, filter ceiling/floor, project to 2D,
    # remove isolated points, inflate obstacles to get occupancy map, etc.
    # IMPORTANT: return map_img as float in value range [0, 1] for visualization downstream.
    # NOTE: in habitat sim, x z plane corresponds to world horizontal plane, and y is vertical.

    # 1. remove ceiling and floor by height
    # - real world is ~2.5m height per floor, 
    #   but the data contains ~4.0m height due to the floor
    # - data seems to be collected from a sensor at height 1.5m from the floor
    #   such that min(data_y) = -1.77m and max(data_y) = 2.365m
    # - in habitat sim, the agent is a 0.1m-radius pillar, and the sensor height
    #   is 1.5m from the floor, so we assume the robot has a physical height
    #   of 1.5 + 0
    # - so, cut floor < 1.3m and ceiling > 0.0m
    ceiling_threshold = 0.0
    floor_threshold = -1.3

    mask_ceiling = coords[:, 1] > ceiling_threshold
    mask_floor = coords[:, 1] < floor_threshold
    mask_remove = mask_ceiling | mask_floor
    
    coords_cut = coords[~mask_remove]
    colors_cut = colors[~mask_remove]
    
    # 2. project to 2D
    # - map (x, z) to (u, v)
    coords_cut_2d = coords_cut[:, [0, 2]]
    
    # - The data itself: 
    #   - min_x = -0.77883, max_x = 0.15992
    #   - min_z = -0.12620, max_z = 0.25277       
    min_u = np.min(coords_cut_2d[:, 0])
    max_u = np.max(coords_cut_2d[:, 0])
    min_v = np.min(coords_cut_2d[:, 1])
    max_v = np.max(coords_cut_2d[:, 1])
    
    # align top-left to origin
    coords_cut_2d[:, 0] -= min_u
    coords_cut_2d[:, 1] -= min_v

    # 3. create map image
    range_u = max_u - min_u
    range_v = max_v - min_v
    map_width = int(range_u * MAP_RESOLUTION) + 1
    map_height = int(range_v * MAP_RESOLUTION) + 1
    
    map_img = np.zeros((map_height, map_width, 3), dtype=np.float32) # pure black background
    pixel_u = (coords_cut_2d[:, 0] * MAP_RESOLUTION).astype(int)
    pixel_v = (coords_cut_2d[:, 1] * MAP_RESOLUTION).astype(int)
    map_img[pixel_v, pixel_u] = colors_cut
    
    # 4. remove isolated points
    occupancy_mask = np.any(map_img > 0, axis=-1).astype(np.uint8)
    
    # find connected color blocks and filter out small noise
    num_labels, labels, stats, centroids = \
        cv2.connectedComponentsWithStats(
            occupancy_mask,
            connectivity=8  # 8: a 3x3 grid centered at the pixel
                # vs. 4: up/down/left/right neightbours
        )
    
    clean_mask = np.zeros_like(occupancy_mask)
    for i in range(1, num_labels): # start from 1, skip 0 (background)
        if stats[i, cv2.CC_STAT_AREA] >= 7: # remove area < 7
            clean_mask[labels == i] = 1

    # make the background + removed noises white
    map_img[clean_mask == 0] = 1.0

    # 5. add patches
    patches = [(159, 87, 160, 100), (91, 36, 93, 47)]
    for u1, v1, u2, v2 in patches:
        # draw lines
        cv2.line(map_img, (u1, v1), (u2, v2), (0, 0, 0), 1)
        cv2.line(clean_mask, (u1, v1), (u2, v2), 1, 1)


    # 6. inflate obstacles to get occupancy map
    robot_radius_m = 0.20 # the agent is a 0.1m-radius pillar by default, bigger for safety
    inflate_pixels = int(robot_radius_m * MAP_RESOLUTION)
    kernel_size = inflate_pixels * 2 + 1
    kernel_inflate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    occupancy_map = cv2.dilate(clean_mask, kernel_inflate, iterations=1)

    # extra edition: 
    # + add a visible inflation
    # + convert RGB to BGR for cv2
    map_img_for_selection = map_img.copy()
    map_img_for_selection[clean_mask == 0] = np.stack([1 - 0.33 * occupancy_map] * 3, axis=-1)[clean_mask == 0]
    map_img_for_selection = map_img_for_selection[:, :, [2, 1, 0]]

    # return translation constants for TODO4 (pixel to world coordinate)
    return map_img, occupancy_map.astype(np.float32), map_img_for_selection, (min_u, min_v, MAP_RESOLUTION)


def select_start(map_img: np.ndarray) -> Tuple[int, int]:
    """Display map and return user-clicked start coordinate."""
    start_point = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            start_point.append((x, y))
            print(f"Start selected: ({x}, {y})")

    cv2.namedWindow("Select Start")
    cv2.setMouseCallback("Select Start", mouse_callback)
    print("Click on the map window to select a start location...")

    while True:
        cv2.imshow("Select Start", (map_img * 255).astype(np.uint8))
        key = cv2.waitKey(1) & 0xFF
        if start_point:
            break
        if key == ord("q"):
            raise RuntimeError("No start selected. Exiting.")

    cv2.destroyWindow("Select Start")
    return start_point[0]


def get_goal_pixels(map_img: np.ndarray, semantic_dict: dict, goal_name: str) -> List[Tuple[int, int]]:
    """function to find all pixels corresponding to the goal object based on color matching."""

    if goal_name.lower() not in semantic_dict:
        raise ValueError(f"Unknown semantic object: {goal_name}. Available options: {list(semantic_dict.keys())}")

    goal_colors = semantic_dict[goal_name.lower()]
    goal_pixels: List[Tuple[float, float]] = []

    for gc in goal_colors:
        gc_norm = np.array(gc) / 255.0
        mask_goal = np.all(np.isclose(map_img, gc_norm, atol=10/255.0), axis=2)
        zs, xs = np.where(mask_goal)
        goal_pixels.extend(list(zip(xs, zs)))

    if not goal_pixels:
        raise ValueError(f"No valid pixels found for '{goal_name}'.")

    return goal_pixels