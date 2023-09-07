from segmentation_module import SAMREALSENSE


sam = SAMREALSENSE()
pointcloud = sam.segment()

print(pointcloud)