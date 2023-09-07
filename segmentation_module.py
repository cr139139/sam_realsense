import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import copy
from mobile_sam import sam_model_registry, SamPredictor
import torch


class SAMREALSENSE:
    def __init__(self):
        self.sam_checkpoint = "./MobileSAM/weights/mobile_sam.pt"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mobile_sam = sam_model_registry["vit_t"](checkpoint=self.sam_checkpoint)
        self.mobile_sam.to(device=self.device)
        self.mobile_sam.eval()
        self.predictor = SamPredictor(self.mobile_sam)

        self.view_ind = 0
        self.breakLoopFlag = 0
        self.backgroundColorFlag = 0

        self.align = rs.align(rs.stream.color)
        self.pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

        profile = self.pipeline.start(config)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

        cv2.namedWindow("Stream", cv2.WINDOW_AUTOSIZE)
        self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(intr.width, intr.height, intr.fx, intr.fy,
                                                                          intr.ppx,
                                                                          intr.ppy)

        self.geometrie_added = False
        self.vis = o3d.visualization.VisualizerWithKeyCallback()

        self.vis.create_window("Pointcloud")
        self.pointcloud = o3d.geometry.PointCloud()
        self.mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        self.mesh_frame.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.vis.register_key_callback(ord("Q"), self.breakLoop)
        self.vis.register_key_callback(ord("K"), self.change_background_color)
        self.mask_index = 0

    def breakLoop(self, vis):
        self.breakLoopFlag += 1
        return False

    def change_background_color(self, vis):
        opt = self.vis.get_render_option()
        if self.backgroundColorFlag:
            opt.background_color = np.asarray([0, 0, 0])
            self.backgroundColorFlag = 0
        else:
            opt.background_color = np.asarray([1, 1, 1])
            self.backgroundColorFlag = 1
        return False

    def segment(self):
        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)

                color_frame = aligned_frames.get_color_frame()
                color_image = np.asanyarray(color_frame.get_data())
                depth_frame = aligned_frames.get_depth_frame()
                depth_color_frame = rs.colorizer().colorize(depth_frame)
                depth_color_image = np.asanyarray(depth_color_frame.get_data())

                depth_image = np.asanyarray(depth_frame.get_data())
                depth_image[depth_image < 10] = 0
                depth_image[depth_image > 1000] = 0
                color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

                h, w, _ = color_image.shape
                self.predictor.set_image(color_image)
                masks, scores, _ = self.predictor.predict(point_coords=np.array([[w // 2, h // 2]]), point_labels=np.array([1]))
                depth_image[~masks[self.mask_index]] = 0
                depth_color_image_masked = copy.deepcopy(depth_color_image)
                depth_color_image_masked[~masks[self.mask_index], :] = 0

                masked_image1 = color_image * masks[0][:, :, np.newaxis]
                masked_image2 = color_image * masks[1][:, :, np.newaxis]
                masked_image3 = color_image * masks[2][:, :, np.newaxis]

                depth = o3d.geometry.Image(depth_image)
                color = o3d.geometry.Image(color_image)

                color_image = cv2.circle(color_image, (w // 2, h // 2), radius=10, color=(0, 0, 255), thickness=-1)
                image1 = np.concatenate([color_image, depth_color_image, depth_color_image_masked], axis=1)
                image2 = np.concatenate([masked_image1, masked_image2, masked_image3], axis=1)
                image = np.concatenate([image1, image2], axis=0)
                image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
                cv2.imshow('Stream', image)

                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth)
                pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.pinhole_camera_intrinsic)
                pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
                pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=0.5)

                if not pcd:
                    continue

                self.pointcloud.clear()
                self.pointcloud += pcd

                if not self.geometrie_added:
                    self.vis.add_geometry(self.pointcloud)
                    self.vis.add_geometry(self.mesh_frame)
                    self.geometrie_added = True

                self.vis.update_geometry(self.pointcloud)
                self.vis.poll_events()
                self.vis.update_renderer()

                key = cv2.waitKey(1)

                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    self.vis.destroy_window()
                    break
                elif key == ord('1'):
                    self.mask_index = 0
                elif key == ord('2'):
                    self.mask_index = 1
                elif key == ord('3'):
                    self.mask_index = 2

                if self.breakLoopFlag:
                    cv2.destroyAllWindows()
                    self.vis.destroy_window()
                    break

        finally:
            self.pipeline.stop()
        return self.pointcloud
