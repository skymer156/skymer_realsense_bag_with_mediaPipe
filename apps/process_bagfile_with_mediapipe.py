# project config loading
import datetime
from config import config

# exception
import sys
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
# Import mediapipe library
import mediapipe as mp

# need to change
# TODO alignment between depth and color frame
# TODO implement 2d pose estimation library (poseNet, MediaPipe, OpenPose, SkeletonTracking)
# TODO 2d point to 3D camera coordinates converting
# TODO judge configulation following realsense type
# TODO create GUI tools
# TODO convert into module
# TODO postprocess processing regarding realsense type
# TODO json or csv streaming


class BagfileEndException(Exception):

    def __init__(self, message) -> None:
        self.message = message
        pass

    def __str__(self) -> str:
        return f"Bagfile End. origin exception : [{self.message}]"


def clipping_background():
    raise NotImplementedError


def setup_logger(name, logfile='LOGFILENAME.log'):
    from logging import getLogger, DEBUG, INFO, FileHandler, StreamHandler, Formatter
    logger = getLogger(name)
    logger.setLevel(DEBUG)

    # create file handler which logs even DEBUG messages
    fh = FileHandler(logfile)
    fh.setLevel(INFO)
    fh_formatter = Formatter(
        '%(asctime)s - %(levelname)s - %(filename)s - %(name)s - %(funcName)s - %(message)s')
    fh.setFormatter(fh_formatter)

    # create console handler with a INFO log level
    ch = StreamHandler()
    ch.setLevel(DEBUG)
    ch_formatter = Formatter(
        '%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
    ch.setFormatter(ch_formatter)

    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def main():
    # 疎結合にしたいので、
    
    # config dcitionary
    # TODO 依存性あり。
    cfg = config.dic_config

    # get src folder path with glob
    bagfile_folder_path = glob.glob(cfg["source_folder"])

    # get logger instance
    logger = setup_logger(__name__, './log/process_for_videostream.log')

    # if bagfile path doesn't exist, output alert and stop this process.
    if not bagfile_folder_path:
        logger.warning("file doesn't exist.")
        logger.warning("input files into apps/src/ folder")
        exit()

    # if source file is not bagfile or bagfile doesnt exist,
    # close this program
    if os.path.splitext(bagfile_folder_path[0])[1] != ".bag":
        logger.warning(".bag file dont exist in apps/src/ folder.")
        logger.warning("Please input bagfile in that folder.")
        exit()

    # for csv header
    # coord = ['x', 'y', 'z']
    # header = ['joint' + str(i//3) + '_' + str(coord[i % 3])
    #          for i in range(cfg["bone_number"] * len(coord))]
    # header.append('framecount')
    # header.append('timedelta[ms]')
    # header.append('UNIXtimestamp[ms]')
    # header.append('count')
    # header.append('datetime')
    # header.append('millis')

    # get now date and get csvname from datetime
    dt = datetime.datetime.now()

    # mediapipe pose instance
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    # pose instance
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    try:
        # create pipeline
        pipeline = rs.pipeline()

        # Create a config object
        rs_cfg = rs.config()

        # TODO need change to enable user to switch processing bagfile
        # TODO change to apply to folder instead file
        # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
        rs.config.enable_device_from_file(
            rs_cfg, bagfile_folder_path[0], repeat_playback=False)

        # Start streaming from file
        profile = pipeline.start(rs_cfg)

        # no real time processing setting
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)

        # get intrinsics
        dpt_intr = rs.video_stream_profile(
            profile.get_stream(rs.stream.depth)).get_intrinsics()
        clo_intr = rs.video_stream_profile(
            profile.get_stream(rs.stream.color)).get_intrinsics()

        logger.debug("depth intrinsics\n"
                    f"intr width : {dpt_intr.width}, intr height : {dpt_intr.height}\n"
                    f"intr fx : {dpt_intr.fx}, intr fy : {dpt_intr.fy}\n"
                    f"intr ppx : {dpt_intr.ppx}, intr ppy : {dpt_intr.ppy}\n")
        logger.debug("color intrinsics\n"
                    f"intr width : {clo_intr.width}, intr height : {clo_intr.height}\n"
                    f"intr fx : {clo_intr.fx}, intr fy : {clo_intr.fy}\n"
                    f"intr ppx : {clo_intr.ppx}, intr ppy : {clo_intr.ppy}\n")

        # depth filter preparing
        thres_fil = rs.threshold_filter(cfg["thres_min"], cfg["thres_max"])

        # Get product line for setting a supporting resolution
        device = profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        logger.debug(f"product name : {device_product_line}")

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break

        if not found_rgb:
            logger.warning("This program need depth camera with color sensor.")
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
            try:
                frames = pipeline.wait_for_frames()
            except Exception as e:
                raise BagfileEndException(f"{e}")

            # Get frame meta data
            framecount = frames.get_frame_number()
            temp = timestamp
            timestamp = frames.get_timestamp()
            delta = timestamp - temp
            backend_timestamp = frames.get_frame_metadata(
                rs.frame_metadata_value.backend_timestamp
            )

            # Print metadata
            logger.debug('time_delta : ' + str(delta) + '[ms]')
            logger.debug('frame_count : ' + str(framecount))
            logger.debug(' timestamp' + str(timestamp))
            logger.debug('backend_timestamp : ' + str(backend_timestamp))
            logger.debug('count ' + f'{count}')

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get depth frame and color frame
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not color_frame or not depth_frame:
                continue

            # count update
            count += 1

            # Colorize depth frame to jet colormap
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            bg_remove_color = color_image.copy()

            # TODO insert LiDAR post processing
            filted_frames = thres_fil.process(depth_frame)
            filted_frames = filted_frames.as_depth_frame()

            # background remove
            depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
            bg_remove_color[
                (depth_image_3d <= clip_distance_forw) |
                (depth_image_3d >= clip_distance_back)] = cfg["fill_color"]

            # pose estimate using background removing image

            results = pose.process(bg_remove_color)
            mp_drawing.draw_landmarks(
                bg_remove_color,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

            # RGB composition converting (RGB to BGR)
            bg_remove_color = cv2.cvtColor(bg_remove_color, cv2.COLOR_RGB2BGR)
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

            # Render images:
            #   depth align to color on left
            #   depth on right
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(
                depth_image, alpha=0.03), cv2.COLORMAP_JET)
            images = np.hstack((bg_remove_color, depth_colormap))

            cv2.imshow("Camera Stream", images)
            key = cv2.waitKey(1)

            # if pressed escape exit program
            if key == 27 or key == ord('q'):
                cv2.destroyAllWindows()
                break

    except BagfileEndException as e:
        logger.warning(f"{e}")

    except Exception as ex:
        logger.warning(
            f"Exception occured: {ex} {type(ex)} {sys.exc_info()[2].tb_lineno}")

    finally:
        pose.close()


if __name__ == '__main__':
    main()
