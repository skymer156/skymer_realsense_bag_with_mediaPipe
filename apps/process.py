# project config loading
from config import config

# First import library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
# Import os.path for file path manipulation
import os.path
# Import glob for fetching bagfile folder series
import glob

# config dcitionary
cfg = config.dic_config


# need to change
#! alignment between depth and color frame
#! implement 2d pose estimation library (poseNet, MediaPipe, OpenPose, SkeletonTracking) 
#! 2d point to 3D camera coordinates converting
#! judge configulation following realsense type
#! create GUI tools
#! convert into module
#! postprocess processing regarding realsense type
#! json or csv streaming
def main():
    # get src folder path with glob
    bagfile_folder_path = glob.glob(cfg["source_folder"])

    if os.path.splitext(bagfile_folder_path[0])[1] != ".bag":
        print(".bag file dont exist in apps/src/ folder.")
        print("Please input bagfile in that folder.")
        exit()

    try:
        # create pipeline
        pipeline = rs.pipeline()

        # Create a config object
        rs_cfg = rs.config()

        # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
        rs.config.enable_device_from_file(rs_cfg, bagfile_folder_path[0])

        # Configure the pipeline to stream the depth stream
        #! Change this parameters according to the recorded bag file resolution
        rs_cfg.enable_stream(rs.stream.depth, rs.format.z16, 30)

        # Start streaming from file
        pipeline.start(rs_cfg)

        # Create openCV window to render image in
        cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)

        # Create colorizer object
        colorizer = rs.colorizer()

        # Streaming loop
        while True:
            # Get frameset of depth
            frames = pipeline.wait_for_frames()

            # Get depth frame
            depth_frame = frames.get_depth_frame()

            # Colorize depth frame to jet colormap
            depth_color_frame = colorizer.colorize(depth_frame)

            # Convert depth_frame to numpy array to render image in opencv
            depth_color_image = np.asanyarray(depth_color_frame.get_data())

            # Render image in opencv window
            cv2.imshow("Depth Stream", depth_color_image)
            key = cv2.waitKey(1)
            # if pressed escape exit program
            if key == 27 or key == ord('q'):
                cv2.destroyAllWindows()
                break

    except Exception:
        print(Exception)
    
    finally:
        pass

