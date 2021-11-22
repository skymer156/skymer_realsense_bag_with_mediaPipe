"""[summary]
project configulations difinition

keys = config_name
value = config_value
"""
dic_config = {
    "source_folder": r"apps/src/*.bag",
    "thres_min": 0.0,
    "thres_max": 4.0,
    "fill_color": 255,
    "fill_min": 0.0,
    "fill_max": 4.0,
    "bone_number": 18,
    "output_folder": r"apps/output/",
    "datetime_format": "%Y_%m_%d_%H_%M_%S",
    "csvname_ext": ".csv",
    "initial_count": 0,
    "initial_timestamp": 0,
    "static_image_mode": True,
    "model_complexity": 2,
    "enable_segmentation": True,
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
    "fps": 30,
    "image_output": r'apps/images/'
}
