from ultralytics import YOLO
from PIL import Image
import cv2
import torch
from PIL import Image
import random
import base64
import io
from Database_Interface import DatabaseInterface
from lan_utils import frontal_face_check
from deepface import DeepFace
import tkinter as tk
from tkinter import simpledialog
import math


db_interface=DatabaseInterface( 'localhost', 'root', 'MySQLRoot', 'faceembeddings')
db_interface.create_tables()

model = YOLO("yolov8-face\yolov8n-face.pt") 


def draw_bounding_box(frame):
    height, width, _ = frame.shape
    
    #bbox_width = int(0.5 * width)
    #bbox_height = int(0.5 * height)
    bbox_width = int(width)
    bbox_height = int(height)
    
    #x = int((width - bbox_width) / 2)
    #y = int((height - bbox_height) / 2)
    x = int((width - bbox_width) / 2)
    y = int((height - bbox_height) / 2)
    
    x2 = x + bbox_width
    y2 = y + bbox_height
    
    cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
    large_bbox = (x, y, x2, y2)

    return large_bbox

def prompt_for_name():
    root = tk.Tk()
    root.withdraw() 
    name = simpledialog.askstring("Input", "Please enter your name:")
    root.destroy()  
    return name


def extract_embeddings(face):
      
    embeddings = DeepFace.represent(face, model_name='Facenet', enforce_detection=False, normalization="Facenet", detector_backend="skip")

    return embeddings



def calculate_bbox_area(bbox):

    x1, y1, x2, y2 = bbox
    
    width = x2 - x1
    height = y2 - y1
    
    area = width * height
    return area




def get_face_img(frame,bbox):
    x1, y1, x2, y2 = map(int, bbox)
    face = frame[y1:y2, x1:x2]
    face_img = Image.fromarray(face)
    return face_img,face



def convert_tensorbytes_2_embeddings(tensor_bytes):

    base64_string = base64.b64encode(tensor_bytes).decode('utf-8')

    decoded_bytes = base64.b64decode(base64_string)

    buffer = io.BytesIO(decoded_bytes)
    buffer.seek(0)
    reconstructed_tensor = torch.load(buffer)

    return reconstructed_tensor



def Match_check(input_embeddings,known_person_data):
    similarities = []
    if len(known_person_data)==0:
        return False
    
    for persons in known_person_data:
        embedding_data=persons[:][2]
        id=persons[:][0]
        name=persons[:][1]
        input_embeddings = torch.tensor(input_embeddings)
        embedding_data = convert_tensorbytes_2_embeddings(embedding_data)
        input_embeddings=input_embeddings.tolist()
        embedding_data=embedding_data.tolist()

        result = DeepFace.verify(input_embeddings,embedding_data, model_name = "Facenet", distance_metric = "euclidean_l2", enforce_detection=False, normalization="Facenet", detector_backend="skip" )
        if result['verified']==True:
            id=id
            print(id)
            return True,name,id

    if result['verified']==False:
        name=None
        id=None
        return False,name,id

def register_face(table,id=None,name=None,binary_embedding=None):

    
    buffer = io.BytesIO()
    torch.save(binary_embedding, buffer)
    buffer.seek(0)
    tensor_bytes = buffer.read()

    db_interface.insert(table, id=id, name=name, embedding=tensor_bytes)
    return True

def store_and_agg_embedding(embeddings, unique_id):

    embeddings=embeddings[0][0]['embedding']
    print(embeddings,"embeddings")

    embeddings=torch.tensor(embeddings)
    uniqid=unique_id
    known_person_data = db_interface.get_known_db()

    match_result=None
    if len(known_person_data)>0:

        match_result,name,id=Match_check(embeddings,known_person_data)

    if len(known_person_data)==0:
        
        text="first_registered {id}"
        id=uniqid

        return text
    if len(known_person_data)>0 and match_result==False:
        id=uniqid
        text=f"not_registered "
        print(f"not registered {id},{name}")
        pname=prompt_for_name()
        register_face("known_person", id, pname, embeddings)
        return text
    elif match_result==True:
        

        text=f"Already_registered {id},{name}"
        print(f"Already registered {id},{name}")
        return text
    else:
        print("error")

def is_bbox_inside(bbox1, bbox2):


    x1_bbox1, y1_bbox1, x2_bbox1, y2_bbox1 = bbox1
    x1_bbox2, y1_bbox2, x2_bbox2, y2_bbox2 = bbox2
    area1=calculate_bbox_area(bbox1)
    area2=calculate_bbox_area(bbox2)
    percent=(area1/area2)*100

    # Check if all four corners of bbox1 are inside bbox2
    if x1_bbox1 >= x1_bbox2 and y1_bbox1 >= y1_bbox2 and x2_bbox1 <= x2_bbox2 and y2_bbox1 <= y2_bbox2 and percent>0.3:
        print("shouldwork")
        return True
    else:
        print("shouldnt_work")
        return 
    
def calculate_centroid(bbox):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return cx, cy

def calculate_distance(centroid1, centroid2):
    print(centroid1,"c1")
    print(centroid2, "c2")
    #cx2, cy2 = centroid2
    #cx1, cy1 = centroid1
    cx1 = centroid1[0].item() if isinstance(centroid1[0], torch.Tensor) else centroid1[0]
    cy1 = centroid1[1].item() if isinstance(centroid1[1], torch.Tensor) else centroid1[1]
    cx2 = centroid2[0].item() if isinstance(centroid2[0], torch.Tensor) else centroid2[0]
    cy2 = centroid2[1].item() if isinstance(centroid2[1], torch.Tensor) else centroid2[1]


    
    return math.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

filename = "pexels_videos_1721303 (1080p)"


vid=rf"C:\Users\joelc\Desktop\python_project\videos\{filename}.mp4"
video_capture = cv2.VideoCapture(0)

output_video_path = 'jllll.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(output_video_path, fourcc, 10.0, (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))

current_track=[] 
frontal_face=[]
bbox=None
text=None
msg=None
coordinates_dict = {}
thresh_dist = 3.0
signal=None

while video_capture.isOpened():

    success, frame = video_capture.read()
    if success:
        draw_bounding_box(frame)
        results = model.track(frame, persist=True, tracker="bytetrack.yaml") 
        bboxes = results[0].boxes.xyxy.cpu()
        conf = results[0].boxes.conf.cpu() 
        print(conf,"cnf")
     
        print(bboxes,"bbx")
        track_ids = results[0].boxes.id 

        if track_ids is not None and current_track in track_ids.tolist():
            cv2.putText(frame, msg, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            for box, track_id in zip(bboxes, track_ids):
                if current_track==track_id:
                   cenbox= box
            centroid=calculate_centroid(cenbox)

            if len(coordinates_dict) <1:
                coordinates_dict[len(coordinates_dict)] = centroid
            elif len(coordinates_dict) < 6 and len(coordinates_dict)>0 :
                ind=len(coordinates_dict)-1
                i=coordinates_dict[ind]
                

                cen=i


                cen_dist=calculate_distance(cen,centroid)
                print(cen_dist,"cen_dist",)
                coordinates_dict[ len(coordinates_dict)] = centroid
                if cen_dist<thresh_dist:
                    coordinates_dict[ len(coordinates_dict)] = centroid
                    print("below threshold")

                else:
                    frontal_face=[]
                    print("above threshold")
                    coordinates_dict={}
            elif len(coordinates_dict)>6:
                coordinates_dict={}
                print("goahead")
                signal="goahead"
        elif track_ids is None:
            current_track=[]
            text=None
            pass

        if track_ids is not None and current_track in track_ids.tolist() and text is not None:
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
           
        if track_ids is None:
            pass
        elif current_track not in track_ids.tolist() and track_ids is not None:
            text=None
            track_ids = results[0].boxes.id.int().cpu().tolist()
            largest_area=0
            largest_area_bbox=0
            largest_area_track_id=0
            for box, track_id in zip(bboxes, track_ids):
                x1, y1, x2, y2 = box
                area=calculate_bbox_area(box)

                if area > largest_area:
                    largest_area = area
                    largest_area_bbox = box
                    largest_area_track_id = track_id
                else:
                    largest_area=largest_area
                    largest_area_bbox = largest_area_bbox
                    largest_area_track_id=largest_area_track_id
            current_track=largest_area_track_id
            bbox=largest_area_bbox
            print(current_track,"curr")

        elif current_track is not None:
            for box, track_id in zip(bboxes, track_ids):
                if current_track==track_id:
                   bbox= box

            pass
        else:
            pass
        print(current_track,"currr")
        annotated_frame = results[0].plot()

    
        if current_track is not None:
            large_bbox= draw_bounding_box(frame)

            if bbox is not None and is_bbox_inside(bbox,large_bbox)==True:

                face_img, face  = get_face_img(frame,bbox)
                face_embed = extract_embeddings(face)
                #print("face_pose")
                face_pose,msg,landmark_count,face_landmarks = frontal_face_check(face)


                if face_pose==True and signal =="goahead":
                    frontal_face.append(face_embed)

                else:
                    frontal_face=[]

                    pass
            else:
                frontal_face=[]
                pass
        else:
            frontal_face=[]
            pass

        if len(frontal_face)==3 and signal =="goahead" and text is None:
            coordinates_dict={}
            signal =None
            front=frontal_face
            print(front,"pr")
            text=None
            unique_id='{:04}'.format(random.randint(1, 9999))
            text=store_and_agg_embedding( front, unique_id)
            frontal_face=[]
            bbox=None
            pass

        else:

            pass 
        cv2.imshow('Frame', annotated_frame)
        output_video.write(annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Frame reading unsuccessful. Exiting loop.")
        break
    

video_capture.release()
output_video.release()

cv2.destroyAllWindows()
