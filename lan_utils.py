
import cv2
import mediapipe as mp
import numpy as np



def frontal_face_check(frame):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5,static_image_mode=False,refine_landmarks=True)
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    frame.flags.writeable = False
    image=cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    results = face_mesh.process(image)
    frame.flags.writeable = True
    img_h, img_w, img_c = frame.shape
    face_3d = []
    face_2d = []
    msg=None
    landmark_count=None
    face_landmark=None
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            face_landmark=face_landmarks.landmark
            landmark_count = len(face_landmarks.landmark)
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])  

            
            # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                    [0, focal_length, img_w / 2],
                                    [0, 0, 1]])

            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            # See where the user's head tilting

            left_eye_center_index = 468
            right_eye_center_index = 473
            if left_eye_center_index >= len(face_landmarks.landmark) or right_eye_center_index >= len(face_landmarks.landmark):
                msg="Eye landmarks not detected"
                face_landmark=None


                return False, msg,landmark_count,face_landmark

            # Extract the left and right eye center coordinates
            left_eye_center = (
                int(face_landmarks.landmark[left_eye_center_index].x * frame.shape[1]),
                int(face_landmarks.landmark[left_eye_center_index].y * frame.shape[0])
            )
            right_eye_center = (
                int(face_landmarks.landmark[right_eye_center_index].x * frame.shape[1]),
                int(face_landmarks.landmark[right_eye_center_index].y * frame.shape[0])
            )
            # Draw center points of both eyes
            cv2.circle(frame, left_eye_center, 5, (0, 255, 0), -1)
            cv2.circle(frame, right_eye_center, 5, (0, 255, 0), -1)

            # Calculate slope of the line passing through both eye centers
            if right_eye_center[0] - left_eye_center[0] != 0:
                slope = (right_eye_center[1] - left_eye_center[1]) / (right_eye_center[0] - left_eye_center[0])
            else:
                slope = float('inf')  # Handle vertical line case

            # Define slope threshold for parallel line
            slope_threshold = 0.1

            # Check if the slope is within the threshold
            if abs(slope) < slope_threshold:
                #self.face_img = self.get_face_image(image_, face_landmarks)
                # cv2.putText(image, 'Face is straight', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                #self.embedding_event.set()
                msg="is_alligned"

                return True,msg,landmark_count,face_landmark
            else:

                msg="is_notalligned"
                return False,msg,landmark_count,face_landmark
                #cv2.putText(image, 'Face is tilted make it straight', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        msg = "No face detected"
        return False, msg, 0, None