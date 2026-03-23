import cv2
import numpy as np

points = []

class Projection(object):

    def __init__(self, image_path, points):
        """
            :param points: Selected pixels on top view(BEV) image
        """

        if type(image_path) != str:
            self.image = image_path
        else:
            self.image = cv2.imread(image_path)
        self.height, self.width, self.channels = self.image.shape
        self.old_pixels = points # Added outside the TODO zone

    def top_to_front(self, theta=-90, phi=0, gamma=0, dx=0, dy=1.5, dz=0, fov=90):
        """
            Project the top view pixels to the front view pixels.
            :return: New pixels on perspective(front) view image

            Assumption: 
            - In the top view, the floor is 2.5 units from the camera. This is 
              given the spec of camera 1.

            The pose of Camera 2 (top) relative to Camera 1 (front) is in params.
            - I want to conceptually treat the camera 1 as the origin (0, 0, 0) 
              facing (0, 0, 0) , and camera 2 is at <dx, dy, dz> turning
              <theta, phi, gamma>
            - In the spec, the camera 2 is at (0, 2.5, 0) turning (-90deg, 0, 0), 
              and camera 1 is at (0, 1, 0) turning (0, 0, 0). So, I set the 
              relative params as offset <0, 1.5, 0> and angular displacement 
              <-90deg, 0, 0>
        """

        # Step 1: revert the points on top-view to world coordinates
        old_pixels = np.array(self.old_pixels)

        # - from [0~511, 0~511] to [-2.5~2.5, -2.5~2.5]
        old_point_offsets = old_pixels / 512.0 * 5.0 - 2.5
        old_point_offsets[:, 1] *= -1

        # - insert z-coordinate such that (x, y) -> (x, y, -2.5)
        points_relToCam2 = np.insert(old_point_offsets, 2, -2.5, axis=1)

        # - apply reverse rotation from pitch(theta), yaw(phi), and roll(gamma)
        rad_x = np.deg2rad(theta)
        rad_y = np.deg2rad(phi)
        rad_z = np.deg2rad(gamma)

        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(rad_x), -np.sin(rad_x)],
            [0, np.sin(rad_x),  np.cos(rad_x)]
        ])
        
        R_y = np.array([
            [np.cos(rad_y), 0, np.sin(rad_y)],
            [0, 1, 0],
            [-np.sin(rad_y), 0, np.cos(rad_y)]
        ])
        
        R_z = np.array([
            [np.cos(rad_z), -np.sin(rad_z), 0],
            [np.sin(rad_z),  np.cos(rad_z), 0],
            [0, 0, 1]
        ])

        R = R_x @ R_y @ R_z

        # - get the points relative to camera 1 by applying translation AFTER rotation
        points_rotated = points_relToCam2 @ R.T
        points_relToCam1 = points_rotated + np.array([dx, dy, dz])

        print(points_relToCam2)
        print(points_rotated)
        print(points_relToCam1)

        # Step 2: mimicking the pin-hole camera with matrix to project the points to front-view

        # - check if any points are behind or exactly on the camera plane 
        #   (Z >= 0, since camera faces -Z)
        if np.any(points_relToCam1[:, 2] >= 0):
            print("\n[WARNING] Some selected points are behind Camera 1! Projection will be distorted.")

        # - mimicking by "divide by -z" and "multiply by f"
        #   NOTE: spec says we have 512x512 output
        f = (512 / 2) / np.tan(np.deg2rad(fov / 2))
        cx, cy = 256.0, 256.0 # 256 = 512 / 2

        u = f * (points_relToCam1[:, 0] / -points_relToCam1[:, 2]) + cx
        v = f * (-points_relToCam1[:, 1] / -points_relToCam1[:, 2]) + cy
        
        new_pixels = np.column_stack((u, v)).astype(int).tolist()

        print(new_pixels)

        return new_pixels

    def show_image(self, new_pixels, img_name='projection.png', color=(0, 0, 255), alpha=0.4):
        """
            Show the projection result and fill the selected area on perspective(front) view image.
        """

        new_image = cv2.fillPoly(
            self.image.copy(), [np.array(new_pixels)], color)
        new_image = cv2.addWeighted(
            new_image, alpha, self.image, (1 - alpha), 0)

        cv2.imshow(
            f'Top to front view projection {img_name}', new_image)
        cv2.imwrite(img_name, new_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return new_image


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        print(x, ' ', y)
        points.append([x, y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img, str(x) + ',' + str(y), (x+5, y+5), font, 0.5, (0, 0, 255), 1)
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('image', img)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:

        print(x, ' ', y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        # cv2.putText(img, str(b) + ',' + str(g) + ',' + str(r), (x, y), font, 1, (255, 255, 0), 2)
        cv2.imshow('image', img)


if __name__ == "__main__":

    pitch_ang = -90

    front_rgb = "bev_data/front2.png"
    top_rgb = "bev_data/bev2.png"

    # click the pixels on window
    img = cv2.imread(top_rgb, 1)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    projection = Projection(front_rgb, points)
    new_pixels = projection.top_to_front(theta=pitch_ang)
    projection.show_image(new_pixels)