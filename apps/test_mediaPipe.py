import mediapipe as mp
import cv2

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

pose = mp_pose.Pose(
    static_image_mode = True,
    min_detection_confidence=0.5,
    model_complexity=2
    )

img = cv2.imread(r"apps/src/sample.jpg")
cv2.imshow("sample image", img)

key = cv2.waitKey(0)

# if pressed escape exit program
if key == 27 or key == ord('q'):
    cv2.destroyAllWindows()

results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

width, height, _ = img.shape

for i in mp_pose.PoseLandmark:
    point_x = results.pose_landmarks.landmark[i].x * width
    point_y = results.pose_landmarks.landmark[i].y * height
    print(str(i),": x {0} , y {1}".format(point_x, point_y))

annotated_image = img.copy()
mp_drawing.draw_landmarks(
    annotated_image,
    results.pose_landmarks,
    mp_pose.POSE_CONNECTIONS,
    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
)

cv2.imshow("pose estimate image",annotated_image)

key = cv2.waitKey(0)

if key == 27 or key == ord('q'):
    cv2.destroyAllWindows()