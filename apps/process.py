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
# Import datetime for calc time
import datetime
# Import csv for save pose data
import csv


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
    # config dcitionary
    cfg = config.dic_config

    # get src folder path with glob
    bagfile_folder_path = glob.glob(cfg["source_folder"])

    # if source file is not bagfile or bagfile doesnt exist,
    # close this program
    if os.path.splitext(bagfile_folder_path[0])[1] != ".bag":
        print(".bag file dont exist in apps/src/ folder.")
        print("Please input bagfile in that folder.")
        exit()

    # for csv header
    coord = ['x', 'y', 'z']
    header = ['joint' + str(i//3) + '_' + str(coord[i % 3])
              for i in range(cfg["bone_number"] * len(coord))]
    header.append('framecount')
    header.append('timedelta[ms]')
    header.append('UNIXtimestamp[ms]')
    header.append('count')
    header.append('datetime')
    header.append('millis')

    # get now date and get csvname from datetime
    dt = datetime.datetime.now()
    csvname = cfg["output_folder"] + \
        dt.strftime(cfg["datetime_format"]) + cfg["csvname_ext"]

    # get file stream for csv file
    #f = open(csvname, 'w', newline='')
    #writer = csv.writer(f)
    
    # write header to csv
    #writer.writerows(header)

    try:
        # create pipeline
        pipeline = rs.pipeline()

        # Create a config object
        rs_cfg = rs.config()

        #! need change to enable user to switch processing bagfile
        # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
        rs.config.enable_device_from_file(
            rs_cfg, bagfile_folder_path[0], repeat_playback=False)

        # Start streaming from file
        profile = pipeline.start(rs_cfg)

        # no real time processing setting
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)

        # get intrinsics
        dpt_intr = rs.video_stream_profile(profile.get_stream(rs.stream.depth)).get_intrinsics()
        clo_intr = rs.video_stream_profile(profile.get_stream(rs.stream.color)).get_intrinsics()

        print("depth intrinsics\n"\
                f"intr width : {dpt_intr.width}, intr height : {dpt_intr.height}\n"\
                f"intr fx : {dpt_intr.fx}, intr fy : {dpt_intr.fy}\n"\
                f"intr ppx : {dpt_intr.ppx}, intr ppy : {dpt_intr.ppy}\n")
        print("color intrinsics\n"\
                f"intr width : {clo_intr.width}, intr height : {clo_intr.height}\n"\
                f"intr fx : {clo_intr.fx}, intr fy : {clo_intr.fy}\n"\
                f"intr ppx : {clo_intr.ppx}, intr ppy : {clo_intr.ppy}\n")
        
        # depth filter preparing
        thres_fil = rs.threshold_filter(cfg["thres_min"], cfg["thres_max"])


        # Get product line for setting a supporting resolution
        device = profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        print(f"product name : {device_product_line}")


        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break

        if not found_rgb:
            print("This program need depth camera with color sensor.")
            exit(0)


        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        align = rs.align(align_to)

        # counter for csv
        framecount = cfg["initial_count"]
        count = cfg["initial_count"]
        timestamp = cfg["initial_timestamp"]


        # calc background cutting value
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        clip_distance_forw = cfg["fill_min"] / depth_scale
        clip_distance_back = cfg["fill_max"] / depth_scale

        cv2.namedWindow("Camera Stream", cv2.WINDOW_AUTOSIZE)

        # Streaming loop
        while True:
            # Get frameset of depth
            frames = pipeline.wait_for_frames()

            # Get frame meta data
            framecount = frames.get_frame_number()
            temp = timestamp
            timestamp = frames.get_timestamp()
            delta = timestamp - temp
            backend_timestamp = frames.get_frame_metadata(
                rs.frame_metadata_value.backend_timestamp
            )

            # Print metadata
            print('time_delta : ' + str(delta) + '[ms]')
            print('frame_count : ' + str(framecount) + 'timestamp' + str(timestamp))
            print('backend_timestamp : ' + str(backend_timestamp))
            #print('datetime :' + ts_date + 'millis' + ts_millis)

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get depth frame and color frame
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not color_frame or not depth_frame:
                continue

            # Colorize depth frame to jet colormap
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            bg_remove_color = color_image.copy()

            #! insert LiDAR post processing
            filted_frames = thres_fil.process(depth_frame)
            filted_frames = filted_frames.as_depth_frame()

            # background remove
            depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
            bg_remove_color[
                (depth_image_3d <= clip_distance_forw) | 
                (depth_image_3d >= clip_distance_back)] = cfg["fill_color"]
            
            # pose estimate using background removing image
            

            # RGB composition converting (RGB to BGR)
            bg_remove_color = cv2.cvtColor(bg_remove_color, cv2.COLOR_RGB2BGR)
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

            # Render images:
            #   depth align to color on left
            #   depth on right
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            images = np.hstack((bg_remove_color, depth_colormap))

            cv2.imshow("Camera Stream", images)
            key = cv2.waitKey(1)

            # if pressed escape exit program
            if key == 27 or key == ord('q'):
                cv2.destroyAllWindows()
                break

    except Exception as ex:
        print(f"Exception occured: \'{ex}\'")

    finally:
        pass
