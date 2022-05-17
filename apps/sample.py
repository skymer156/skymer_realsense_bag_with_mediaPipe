from email import header
import mediapipe as mp

import csv

from process_bagfile_with_mediapipe import generate_csv_header

mp_pose = mp.solutions.pose

axis = ['x', 'y', 'z']
header = []

# for i in list(mp_pose.PoseLandmark):
#     landmarkname = str(i).split('.')[1]
#     landmarkAxis = [f'{landmarkname}_{ax}' for ax in axis]
#     print(landmarkAxis)
#     header.extend(landmarkAxis)

with open('sample.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(header)

landmarklist = [ str(i).split('.')[1] for i in list(mp_pose.PoseLandmark) ]
header = generate_csv_header(landmarklist, axis)

print(header)