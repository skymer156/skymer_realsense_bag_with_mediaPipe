"""[summary]
project configulations difinition

keys = config_name
value = config_value
"""
dic_config = {
    "source_folder": r"C:\Users\yota0\Documents\Yota\program\python_proj\MediaPipe\skeletonTracking_Lider\apps\src\*.bag",
    "thres_min": 0.0,
    "thres_max": 4.0,
    "fill_color": 255,
    "fill_min": 0.0,
    "fill_max": 4.0,
    "bone_number": 18,
    "output_folder": r"csvdata/",
    "datetime_format": "%Y_%m_%d_%H_%M_%S",
    "csvname_ext": ".csv",
    "initial_count": 0,
    "initial_timestamp": 0,
    "static_image_mode": True,
    "model_complexity": 2,
    "enable_segmentation": True,
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5
}
