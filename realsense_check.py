import pyrealsense2 as rs
import numpy as np
import cv2
import time
import open3d as o3d
import os

view_ind = 0
breakLoopFlag = 0
backgroundColorFlag = 1


def breakLoop(vis):
    global breakLoopFlag
    breakLoopFlag += 1
    return False


def change_background_color(vis):
    global backgroundColorFlag
    opt = vis.get_render_option()
    if backgroundColorFlag:
        opt.background_color = np.asarray([0, 0, 0])
        backgroundColorFlag = 0
    else:
        opt.background_color = np.asarray([1, 1, 1])
        backgroundColorFlag = 1
    # background_color ~=backgroundColorFlag
    return False


align = rs.align(rs.stream.color)
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

profile = pipeline.start(config)

intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Color Stream", cv2.WINDOW_AUTOSIZE)
pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(intr.width, intr.height, intr.fx, intr.fy, intr.ppx,
                                                             intr.ppy)

geometrie_added = False
vis = o3d.visualization.VisualizerWithKeyCallback()

vis.create_window("Pointcloud")
pointcloud = o3d.geometry.PointCloud()

vis.register_key_callback(ord("Q"), breakLoop)
vis.register_key_callback(ord("K"), change_background_color)

try:
    while True:
        time_start = time.time()

        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        depth_frame = aligned_frames.get_depth_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        depth_image[depth_image < 10] = 0
        depth_image[depth_image > 1000] = 0
        color_image1 = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        cv2.imshow('Color Stream', color_image1)

        depth = o3d.geometry.Image(depth_image)
        color = o3d.geometry.Image(color_image)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=5, std_ratio=1.0)

        if not pcd:
            continue

        pointcloud.clear()
        pointcloud += pcd

        if not geometrie_added:
            vis.add_geometry(pointcloud)
            geometrie_added = True

        vis.update_geometry(pointcloud)
        vis.poll_events()
        vis.update_renderer()

        depth_color_frame = rs.colorizer().colorize(depth_frame)

        depth_color_image = np.asanyarray(depth_color_frame.get_data())

        cv2.imshow("Depth Stream", depth_color_image)
        key = cv2.waitKey(1)

        time_end = time.time()
        # print("FPS = {0}".format(int(1/(time_end - time_start))))

        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            vis.destroy_window()
            break

        if breakLoopFlag:
            cv2.destroyAllWindows()
            vis.destroy_window()
            break

finally:
    pipeline.stop()
