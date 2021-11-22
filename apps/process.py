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
# Import mediapipe library
import mediapipe as mp
# Import itertool for process iterator
import itertools
# Import traceback for debug
import traceback


# need to change
# alignment between depth and color frame
# implement 2d pose estimation library (poseNet, MediaPipe, OpenPose, SkeletonTracking)
# 2d point to 3D camera coordinates converting
# judge configulation following realsense type
# create GUI tools
# convert into module
# postprocess processing regarding realsense type
# json or csv streaming
# extract magic number
# add folder check function
# get ndarray memory with estimate frame

def clipping_background():
    raise NotImplementedError


def width_clip(img_width, value):
    if value <= 0:
        return 0
    elif value >= img_width:
        return img_width - 1
    return value


def height_clip(img_height, value):
    if value < 0:
        return 0
    elif value >= img_height:
        return img_height - 1
    return value


def is_inshape(width: int, height: int, pixels) -> bool:
    bool_inshape = True
    if not 0 <= pixels[0] < width:
        bool_inshape = False
    if not 0 <= pixels[1] < height:
        bool_inshape = False
    return bool_inshape


def main():
    # config dcitionary
    cfg = config.dic_config

    # get src folder path with glob
    bagfile_folder_path = glob.glob(cfg["source_folder"])

    # if bagfile path doesn't exist, output alert and stop this process.
    if not bagfile_folder_path:
        print("file doesn't exist.")
        print("input files into apps/src/ folder")
        exit()

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
    f = open(csvname, 'w', newline='')
    writer = csv.writer(f, lineterminator='\n')

    # write header to csv
    # writer.writerows(header)

    try:
        # create pipeline
        pipeline = rs.pipeline()

        # Create a config object
        rs_cfg = rs.config()

        #! need change to enable user to switch processing bagfile
        #! change to apply to folder instead file
        # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
        rs.config.enable_device_from_file(
            rs_cfg, bagfile_folder_path[2], repeat_playback=False)

        # Start streaming from file
        profile = pipeline.start(rs_cfg)

        # no real time processing setting
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)
        duration = playback.get_duration()
        estimate_frame_length = cfg["fps"] * \
            (duration.seconds + duration.microseconds / 1.0e6)
        print(f"bagfile estimate frame length is {estimate_frame_length}")
        print(f"bagfile total time is {duration}")

        # get intrinsics
        dpt_intr = rs.video_stream_profile(
            profile.get_stream(rs.stream.depth)).get_intrinsics()
        clo_intr = rs.video_stream_profile(
            profile.get_stream(rs.stream.color)).get_intrinsics()

        print("depth intrinsics\n"
              f"intr width : {dpt_intr.width}, intr height : {dpt_intr.height}\n"
              f"intr fx : {dpt_intr.fx}, intr fy : {dpt_intr.fy}\n"
              f"intr ppx : {dpt_intr.ppx}, intr ppy : {dpt_intr.ppy}\n")
        print("color intrinsics\n"
              f"intr width : {clo_intr.width}, intr height : {clo_intr.height}\n"
              f"intr fx : {clo_intr.fx}, intr fy : {clo_intr.fy}\n"
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
        count = -1
        timestamp = cfg["initial_timestamp"]

        # calc background cutting value
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        clip_distance_forw = cfg["fill_min"] / depth_scale
        clip_distance_back = cfg["fill_max"] / depth_scale

        cv2.namedWindow("Camera Stream", cv2.WINDOW_AUTOSIZE)

        # mediapipe pose instance
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_pose = mp.solutions.pose

        # pose instance
        pose = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        datalist_for_csv = []

        # Streaming loop
        while True:
            # Get frameset of depth
            frames = pipeline.wait_for_frames()
            count += 1

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
            print('frame_count : ' + str(framecount) +
                  'timestamp ' + str(timestamp))
            print('backend_timestamp : ' + str(backend_timestamp))
            #print('datetime :' + ts_date + 'millis' + ts_millis)

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get depth frame and color frame
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not color_frame or not depth_frame:
                print("each frame doesnt exist!!")
                continue

            # Colorize depth frame to jet colormap
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            bg_remove_color = color_image.copy()

            # need to insert LiDAR post processing
            filted_frames = thres_fil.process(depth_frame)
            filted_frames = filted_frames.as_depth_frame()

            # background remove
            depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
            bg_remove_color[
                ((depth_image_3d <= clip_distance_forw) |
                 (depth_image_3d >= clip_distance_back))
            ] = cfg["fill_color"]

            # RGB composition converting (RGB to BGR)
            # bg_remove_color = cv2.cvtColor(bg_remove_color, cv2.COLOR_RGB2BGR)
            # color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

            # pose estimate using background removing image
            results = pose.process(color_image)

            # pose estimation check
            if not results.pose_landmarks:
                print("not detected!!")
                #! not implemented. i want exception csv writing if not pose estimate.
                continue

            # get image width and height
            height, width, _ = color_image.shape

            # get xy pixel : list in list [ [x, y (joint)] each joint  ]
            joint_xy_pixels = [
                [
                    round(results.pose_landmarks.landmark[joint].x * width),
                    round(results.pose_landmarks.landmark[joint].y * height)
                ] for joint in mp_pose.PoseLandmark
            ]

            # cliped joints xy pixels generator (not in memory)
            clip_xy_pixels = (
                [width_clip(width, xy[0]), height_clip(height, xy[1])]
                for xy in joint_xy_pixels
            )

            # depth value generator in joint pixel
            depth_value = (
                filted_frames.get_distance(
                    pixel[0], pixel[1]
                ) for pixel in clip_xy_pixels
            )

            # realsense 3D points in camera coordinates
            points = (
                rs.rs2_deproject_pixel_to_point(
                    clo_intr, joint_xy_pixels[i], depth
                ) for i, depth in enumerate(depth_value)
            )

            # if joint isn't in image area, convert point [-1,-1,-1]
            points_iter = itertools.chain.from_iterable(
                (point if is_inshape(width, height, joint_xy_pixels[i]
                                     ) else [-1, -1, -1]
                 ) for i, point in enumerate(points)
            )

            # datalist_for_csv.append(list(points_iter))
            writer.writerow(list(points_iter))

            """ for joint in mp_pose.PoseLandmark:
                print(
                    f"{str(joint)} : x {next(points_iter)} y {next(points_iter)} z {next(points_iter)}") """

            # Render images:
            #   depth align to color on left
            #   depth on right
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                color_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=1), cv2.COLORMAP_JET)
            images = np.hstack((color_image, depth_colormap))

            cv2.imshow("Camera Stream", images)
            key = cv2.waitKey(1)

            # if pressed escape exit program
            if key == 27 or key == ord('q'):
                cv2.destroyAllWindows()
                break

    except Exception as ex:
        print(f"Exception occured: \'{ex}\'")
        traceback.print_exc()
        writer.close()

    finally:
        pose.close()
        pass
