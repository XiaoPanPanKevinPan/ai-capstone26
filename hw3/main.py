from map_processor import MAP_RESOLUTION
import random
import sys
from typing import List, Tuple
import cv2
import numpy as np

from map_processor import load_and_filter_map, select_start, get_goal_pixels
from navigator import init_sim, execute_waypoint_path


POINT_CLOUD_DATA = "semantic_3d_pointcloud/point.npy"
COLOR_DATA = "semantic_3d_pointcloud/color01.npy"

# Sample semantic color and index dictionaries for a few object categories. 
# Check hw0/replica_v1/apartment_0/habitat/info_semantic.json and 
# hw3/color_coding_semantic_segmentation_classes.xlsx for the full list of 
# categories and their corresponding colors and indices.
SEMANTIC_DICTS = {
    "colors": {
        "rack": [[0, 255, 133]],
        "cooktop": [[7, 255, 224]],
        "sofa": [[10, 0, 255]],
        "cushion": [[255, 9, 92]],
        "stair": [[173, 255, 0]]
    },
    "indices": {
        "rack": 8,
        "cooktop": 280,
        "sofa": 196,
        "cushion": 430,
        "stair": 192
    },
}


def pick_goal(map_img) -> Tuple[str, Tuple[int, int]]:
    prompt = "Enter semantic destination (ex: 'rack', 'cooktop', 'sofa'): "
    goal_prompt = input(prompt).strip().lower()
    if goal_prompt not in SEMANTIC_DICTS["colors"]:
        print(f"Goal '{goal_prompt}' is not valid.")
        sys.exit(1)

    goal_pixels = get_goal_pixels(map_img, SEMANTIC_DICTS["colors"], goal_prompt)
    goal = random.choice(goal_pixels)
    return goal_prompt, goal


def run_in_sim(start_world: Tuple[float, float], world_path: List[Tuple[float, float]], goal_prompt: str):
    start_x, start_z = start_world
    print(f"Spawning Agent at world position: ({start_x:.3f}, {start_z:.3f})")

    sim, agent, _ = init_sim(start_x=start_x, start_z=start_z)

    id = SEMANTIC_DICTS["indices"][goal_prompt]
    # abandoned since the coordination mapping is weird

    # if id is None:
    #     # load "../hw0/replica_v1/apartment_0/habitat/info_semantic.json"
    #     import json
    #     import math
    #     try:
    #         with open("../hw0/replica_v1/apartment_0/habitat/info_semantic.json", "r") as f:
    #             info = json.load(f)
            
    #         goal_x, goal_z = world_path[-1] # the "new_goal"
    #         min_dist = float('inf')
    #         best_id = None
            
    #         for obj in info.get("objects", []):
    #             if obj.get("class_name") != goal_prompt:
    #                 continue

    #             center = obj.get("oriented_bbox", {}) \
    #                 .get("abb", {}) \
    #                 .get("center", None)
    #             sizes = obj.get("oriented_bbox", {}) \
    #                 .get("abb", {}) \
    #                 .get("sizes", None)

    #             if not center or not sizes:
    #                 continue 

    #             # 計算 2D 平面距離 Bounding Box 邊緣的最短距離
    #             # 如果已經在 Box 內部，則 dx, dz 為 0，dist 為 0
    #             # Note: [a, b, c] in JSON should map to [x, y, z] = [-a, c, -b]
    #             center_in_world = [-center[0], center[2], -center[1]]
    #             dx = max(0, abs(goal_x - center_in_world[0]) - sizes[0] / 2)
    #             dz = max(0, abs(goal_z - center_in_world[2]) - sizes[2] / 2)
    #             dist = math.hypot(dx, dz)
                
    #             if dist < min_dist:
    #                 min_dist = dist
    #                 best_id = int(obj["id"])
                            
    #         if best_id is not None and min_dist < 1.0: # 1m
    #             id = best_id
    #             print(f"Dynamically resolved '{goal_prompt}' to instance ID: {id} (distance: {min_dist:.3f}m from goal point)")
    #         else:
    #             print(f"Error: Could not find corresponding '{goal_prompt}' in info_semantic.json")
    #             print(f"So far the best is {best_id} with distance {min_dist:.3f}m")
    #             print(f"Goal point: ({goal_x}, {goal_z})")
    #             return
    #     except Exception as e:
    #         print(f"Error: Could not dynamically load info_semantic.json: {e}")
    #         return
        
    execute_waypoint_path(world_path, sim, agent, id)


    
def dist(p1, p2):
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))
    
def is_collision_free(p1, p2, occupancy_map):
    height, width = occupancy_map.shape
    d = dist(p1, p2)
    steps = int(d / 1.0) + 1  # sample every 1 pixel
    for i in range(steps + 1):
        t = i / steps if steps > 0 else 0
        u = p1[0] + t * (p2[0] - p1[0])
        v = p1[1] + t * (p2[1] - p1[1])
        iu, iv = int(round(u)), int(round(v))
        if iu < 0 or iu >= width or iv < 0 or iv >= height:
            return False
        if occupancy_map[iv, iu] > 0.5: # > 0.5 means obstacle
            return False
    return True

# Note: Goal is a directional hint. 
#       We should stop once the surrounding meets goal_prompt
def plan_path(start, goal_prompt, goal, occupancy_map, map_img):
    MAX_ITER = 50000
    STEP_SIZE = 0.10 * MAP_RESOLUTION   # 1m per step
    GOAL_BIAS = 0.50                    # 50% chance to directly explore towards the goal
    GOAL_DIST = 0.40 * MAP_RESOLUTION   # 1m radius to accept the goal
    
    height, width = occupancy_map.shape
    goal_color = np.array(SEMANTIC_DICTS["colors"][goal_prompt][0]) / 255.0
        # the sementic color of the goal

    # preallocate np array as the tree
    tree_arr = np.zeros((MAX_ITER + 2, 2), dtype=float)
    tree_arr[0] = start
    num_nodes = 1
    
    # dict for parents (node index : parent node index)
    parents = {0: None}
    
    print("Running RRT...")
    for i in range(MAX_ITER):
        # 1. random sampling
        if random.random() < GOAL_BIAS:
            x_rand = goal
        else:
            x_rand = (random.uniform(0, width - 1), random.uniform(0, height - 1))
            
        # 2. find nearest node
        diffs = tree_arr[:num_nodes] - x_rand
        sq_dists = diffs[:, 0]**2 + diffs[:, 1]**2 # no sqrt makes it faster
        nearest_idx = int(np.argmin(sq_dists))
        x_nearest = tree_arr[nearest_idx]
        
        # 3. step from the nearest node
        d = dist(x_nearest, x_rand)
        if d <= STEP_SIZE:
            x_new = x_rand
        else:
            theta = np.arctan2(x_rand[1] - x_nearest[1], x_rand[0] - x_nearest[0])
            x_new = (
                float(x_nearest[0] + STEP_SIZE * np.cos(theta)),
                float(x_nearest[1] + STEP_SIZE * np.sin(theta))
            )
            
        # 4. collision check
        if not is_collision_free(x_nearest, x_new, occupancy_map):
            continue
        
        # 5. add node to tree array
        new_idx = num_nodes
        tree_arr[new_idx] = x_new
        parents[new_idx] = nearest_idx
        num_nodes += 1
            
        # 6. check if goal is reached 
        # - the goal color should be in a rectangle centered at x_new

        # 6.1. local map (a rectangle)
        u_c, v_c = int(round(x_new[0])), int(round(x_new[1]))
        r = int(GOAL_DIST)
        u_min, u_max = max(0, u_c - r), min(width, u_c + r + 1)
        v_min, v_max = max(0, v_c - r), min(height, v_c + r + 1)
        local_map = map_img[v_min:v_max, u_min:u_max]
        
        # 6.2. check if the goal color is in the local map
        mask_goal = np.all(np.isclose(local_map, goal_color, atol=10/255.0), axis=-1)
        if not np.any(mask_goal):
            continue

        # 6.3. select the goal color in local map as the new goal
        goal_pixels = np.where(mask_goal)
        new_goal = (goal_pixels[1][0] + u_min, goal_pixels[0][0] + v_min)
        
        # 7. once found, stop and form the reverted path
        path = []
        curr = new_idx
        while curr is not None:
            path.append(tuple(tree_arr[curr]))
            curr = parents[curr]
        path.reverse()

        # 7.1. append the new_goal in the end of the path 
        #      (the simulator will try to get the closest no matter reachable or not)
        path.append(new_goal) 
        print(f"RRT Success! Found a path with {len(path)} steps.")

        # 7.2 leave obstacles
        leave_obstacle_path = path_leave_obstacles(path, occupancy_map)
        print(f"Path left obstacles.")

        # 7.3. simplify the path
        simple_path = simplify_path(path, occupancy_map)
        print(f"Path simplified to {len(simple_path)} steps.")

        # 7.4. leave obstacles
        leave_obstacle_simple_path = path_leave_obstacles(simple_path, occupancy_map)
        print(f"Path left obstacles.")

        return leave_obstacle_simple_path, simple_path, leave_obstacle_path, tree_arr[:num_nodes], parents
                    
    return None, None, None, None, None

def simplify_path(path, occupancy_map):
    if len(path) <= 3:
        return path

    core_path = path[:-1] # keep the goal point
    simple_path = [core_path[0]]
    curr_idx = 0
    
    while curr_idx < len(core_path) - 1: # keep the last point to see the object
        # find the furthest node that can be connected to the current node
        for next_idx in range(len(core_path) - 1, curr_idx, -1):
            if is_collision_free(core_path[curr_idx], core_path[next_idx], occupancy_map):
                simple_path.append(core_path[next_idx])
                curr_idx = next_idx
                break
                
    simple_path.append(path[-1])
    return simple_path
    
def path_leave_obstacles(path, occupancy_map):
    if len(path) <= 3:
        return path
        
    height, width = occupancy_map.shape
    new_path = list(path)
    
    # helper: spiral search for nearest obstacle
    def find_nearest_obstacle(p, max_r=50):
        u0, v0 = int(round(p[0])), int(round(p[1]))
        if occupancy_map[min(max(v0, 0), height-1), min(max(u0, 0), width-1)] > 0.5:
            return (u0, v0)

    # helper: spiral search for nearest obstacle
    def find_nearest_obstacle(p, max_r=50):
        u0, v0 = int(round(p[0])), int(round(p[1]))
        if occupancy_map[min(max(v0, 0), height-1), min(max(u0, 0), width-1)] > 0.5:
            return (u0, v0)

        def find_nearest_at_r(r):
            best_dist_sq = float('inf')
            nearest_in_r = None
            for du in range(-r, r+1):
                for dv in [-r, r]:
                    u, v = u0 + du, v0 + dv
                    if 0 <= u < width and 0 <= v < height and occupancy_map[v, u] > 0.5:
                        dist_sq = du**2 + dv**2
                        if dist_sq < best_dist_sq:
                            best_dist_sq = dist_sq
                            nearest_in_r = (du, dv)
            for dv in range(-r+1, r):
                for du in [-r, r]:
                    u, v = u0 + du, v0 + dv
                    if 0 <= u < width and 0 <= v < height and occupancy_map[v, u] > 0.5:
                        dist_sq = du**2 + dv**2
                        if dist_sq < best_dist_sq:
                            best_dist_sq = dist_sq
                            nearest_in_r = (du, dv)
            return nearest_in_r, best_dist_sq
        
        nearest = None
        nearest_r = max_r
        nearest_dist_sq = float('inf')
        
        for r in range(1, max_r):
            pt, dist_sq = find_nearest_at_r(r)
            if pt is not None:
                nearest = pt
                nearest_r = r
                nearest_dist_sq = dist_sq
                break

        if nearest is None:
            return None
        
        # prevent ignoring the nearest obstacle, seach further
        max_search_r = min(max_r, int(nearest_dist_sq ** 0.5) + 1)
        for r in range(nearest_r + 1, max_search_r + 1):
            pt, dist_sq = find_nearest_at_r(r)
            if pt is not None and dist_sq < nearest_dist_sq:
                nearest = pt
                nearest_dist_sq = dist_sq
                
        return (u0 + nearest[0], v0 + nearest[1])

    # check if offset P_curr by t * N, is it collision free to P_prev and P_next
    def check_free(t, P_curr, N, P_prev, P_next):
        P_test = (P_curr[0] + t * N[0], P_curr[1] + t * N[1])
        # if out of map, it's a collision
        if not (0 <= P_test[0] < width and 0 <= P_test[1] < height):
            return False
        return is_collision_free(P_prev, P_test, occupancy_map) and \
               is_collision_free(P_test, P_next, occupancy_map)

    # find a segment containing the original point, and 
    # all points except for the bounaries are collision free
    def boundary_search(P_curr, N, P_prev, P_next, step_dir):
        t = 0.0
        step = 2.0 * step_dir
        
        # 1. exponential expansion: find an upper bound
        while check_free(t + step, P_curr, N, P_prev, P_next):
            t += step
            step *= 2.0
            if abs(t) > 500: # 防呆，不會真的跑無限遠
                break
                
        # 2. binary search: find the exact boundary
        low = t
        high = t + step
        while not abs(high - low) < 1.42: # ensure px level precision
            mid = (low + high) / 2.0
            if check_free(mid, P_curr, N, P_prev, P_next):
                low = mid  # this mid is collision free
            else:
                high = mid # this mid is not collision free
                
        return low

    # do the whole path for five times
    for pass_idx in range(5): 
        # don't move start point and the last two points
        for i in range(1, len(new_path) - 2):
            P_prev = new_path[i-1]
            P_curr = new_path[i]
            P_next = new_path[i+1]
            
            # tengent vector: (prev - next) point
            dx = P_next[0] - P_prev[0]
            dy = P_next[1] - P_prev[1]
            length = np.hypot(dx, dy)
            if length < 1e-3:
                continue
                
            # find the normal vector perpendicular to the tengent vector
            N = (-dy / length, dx / length)
            
            ## 1. move according to norm
            # find the boundaries (happen to be not collision free)
            t_pos = boundary_search(P_curr, N, P_prev, P_next, 1)
            t_neg = boundary_search(P_curr, N, P_prev, P_next, -1)
            
            # take the middle of the two boundaries as the new path point
            opt_t = (t_pos + t_neg) / 2.0
            new_path[i] = (P_curr[0] + opt_t * N[0], P_curr[1] + opt_t * N[1])

            ## 2. move leaving nearest obstacle
            P_curr = new_path[i] # Update P_curr from Step 1
            O1 = find_nearest_obstacle(P_curr)
            if O1 is None:
                continue

            dx_o = P_curr[0] - O1[0]
            dy_o = P_curr[1] - O1[1]
            length_o = np.hypot(dx_o, dy_o)

            # Only push if we are reasonably far from O1 (don't divide by zero)
            if length_o < 1e-3:
                continue

            # same procedure as the above
            M = (dx_o / length_o, dy_o / length_o)
            
            t_pos_o = boundary_search(P_curr, M, P_prev, P_next, 1)
            t_neg_o = boundary_search(P_curr, M, P_prev, P_next, -1)
            
            opt_t_o = (t_pos_o + t_neg_o) / 2.0
            new_path[i] = (P_curr[0] + opt_t_o * M[0], P_curr[1] + opt_t_o * M[1])
            
    return new_path

def main():
    """Entry point."""

    print("=== Step 1: Processing the 3D Map ===")
    # =============== TODO 1-2 ===============
    map_img, occupancy_map, map_img_for_display, trans_info = load_and_filter_map(POINT_CLOUD_DATA, COLOR_DATA)

    print("=== Step 2: Selecting Agent Start and Goal Positions ===")
    start = select_start(map_img_for_display)
    goal_prompt, goal = pick_goal(map_img)
    print(f"Goal pixel selected at coordinates: {goal}")


    print("=== Step 3: Executing Path Planning (RRT) ===")
    # =============== TODO 2 ===============
    # implement RRT path planner in plan_path()

    path, simple_path, leave_obstacle_path, tree, parents = plan_path(start, goal_prompt, goal, occupancy_map, map_img)
    if not path:
        print("Planner could not find a path.")
        sys.exit(1)


    print("=== Step 4: Visualizing the Planned Path ===")
    # =============== TODO 3 ===============
    # Visualize the planned path over the map

    vis_map = (np.copy(map_img_for_display) * 255).astype(np.uint8)
    vis_map = np.ascontiguousarray(vis_map) # prevent cv2 complaining
        
    # draw the RRT tree (light blue) and nodes (cyan)
    for idx in range(1, len(tree)):
        parent_idx = parents[idx]
        if parent_idx is not None:
            pt_end = (int(tree[idx][0]), int(tree[idx][1]))
            pt_start = (int(tree[parent_idx][0]), int(tree[parent_idx][1]))
            cv2.line(vis_map, pt_start, pt_end, (255, 200, 150), 1)
            cv2.circle(vis_map, pt_end, 1, (255, 255, 0), -1)

    # draw the leave obstacle path (blue line) and nodes (light green)
    for i in range(len(leave_obstacle_path) - 1):
        pt1 = (int(leave_obstacle_path[i][0]), int(leave_obstacle_path[i][1]))
        pt2 = (int(leave_obstacle_path[i+1][0]), int(leave_obstacle_path[i+1][1]))
        cv2.line(vis_map, pt1, pt2, (255, 0, 0), 1)
        cv2.circle(vis_map, pt1, 1, (175, 255, 175), -1)

    # draw the simple path (yellow line) and nodes (light green)
    for i in range(len(simple_path) - 1):
        pt1 = (int(simple_path[i][0]), int(simple_path[i][1]))
        pt2 = (int(simple_path[i+1][0]), int(simple_path[i+1][1]))
        cv2.line(vis_map, pt1, pt2, (0, 255, 255), 1)
        cv2.circle(vis_map, pt1, 1, (175, 255, 175), -1)

    # draw the final path (red line) and nodes (green)
    for i in range(len(path) - 1):
        pt1 = (int(path[i][0]), int(path[i][1]))
        pt2 = (int(path[i+1][0]), int(path[i+1][1]))
        cv2.line(vis_map, pt1, pt2, (0, 0, 255), 1)
        cv2.circle(vis_map, pt1, 1, (0, 255, 0), -1)
        
    # draw the goal point (black big circle)
    new_goal = path[-1]
    cv2.circle(vis_map, (int(new_goal[0]), int(new_goal[1])), 2, (0, 0, 0), -1)
    
    cv2.imshow("RRT Found Path", vis_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    print("=== Step 5: Translating Path to Habitat Simulator ===")
    # =============== TODO 4 ===============
    # Convert pixel path to world coordinates
    # world_path is a list of tuples(float, float) representing waypoints in world coordinates
    
    min_u, min_v, map_resolution = trans_info
    world_path = []
    
    for (u, v) in path:
        x_world = (u / map_resolution) + min_u
        z_world = (v / map_resolution) + min_v
        world_path.append((x_world, z_world)) 

    run_in_sim(world_path[0], world_path, goal_prompt)


if __name__ == "__main__":
    main()
