import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import time
import csv
import os
from joblib import load


out = 0
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # bgr is compatible with OpenCV
#pipeline.start(config)
pipelineProfile = pipeline.start(config)

while True:
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    if not depth_frame or not color_frame:
        continue

    color_img = np.asanyarray(color_frame.get_data())
    depth_img = np.asanyarray(depth_frame.get_data())
    color_img1 = color_img.copy()



    
    if out == 0:
        
        profile = depth_frame.get_profile()
        intrinsics = profile.as_video_stream_profile().get_intrinsics()
        depth_profile = frames.get_depth_frame().get_profile().as_video_stream_profile()
        color_profile = frames.get_color_frame().get_profile().as_video_stream_profile()

        extrinsics = depth_profile.get_extrinsics_to(color_profile)
        print(f"extrinsics:\n{extrinsics}")
        # print(f"Rotation: {extrinsics.rotation}")
        print(intrinsics)

    out = 1


    
    x = 320
    y = 240
    dist1 = depth_frame.get_distance(x,y)
   


    coordinate_3d1 = rs.rs2_deproject_pixel_to_point(intrinsics, (x,y), dist1)
    cv2.circle(color_img, (x,y), 10, 255, 2)
    cv2.putText(color_img, f'depth:{dist1:.2f}m | 3D:({coordinate_3d1[0]:.2f}, {coordinate_3d1[1]:.2f}, {coordinate_3d1[2]:.2f})', 
            (x-100, y-40), cv2.FONT_ITALIC, 0.6, (255, 0, 0), 3)
    

    x2 = 100
    y2 = 100
    dist2 = depth_frame.get_distance(x2,y2)
    coordinate_3d2 = rs.rs2_deproject_pixel_to_point(intrinsics, (x2,y2), dist2)
    cv2.circle(color_img, (x2,y2), 10, 255, 2)
    cv2.putText(color_img, f'depth:{dist2:.2f}m | 3D:({coordinate_3d2[0]:.2f}, {coordinate_3d2[1]:.2f}, {coordinate_3d2[2]:.2f})', 
            (x2-100, y2-40), cv2.FONT_ITALIC, 0.6, (255, 0, 0), 3)

    cv2.imshow('Projection', color_img)
    
    cv2.waitKey(1)
    #cv2.destroyAllWindows()