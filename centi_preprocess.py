import sys
import json
from pathlib import Path
import cv2
import numpy as np
from skimage.morphology import skeletonize
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, 
    QMainWindow, 
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QLineEdit,
    QLabel,
    QFileDialog,
    QFormLayout,
    QMessageBox
)

def set_directory_tree():
    Path("output_files").mkdir(parents=True, exist_ok=True)
    Path("output_files/csvs").mkdir(parents=True, exist_ok=True)
    Path("output_files/graphs").mkdir(parents=True, exist_ok=True)
    Path("output_files/heatmaps").mkdir(parents=True, exist_ok=True)
    Path("output_files/videos").mkdir(parents=True, exist_ok=True)
    Path("processed_videos").mkdir(parents=True, exist_ok=True)
    Path("compressed").mkdir(parents=True, exist_ok=True)
    Path("process_values.json").touch(exist_ok=True)
    Path("head.json").touch(exist_ok=True)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("My App")
        self.resize(300, 200)
        self.json_file = "process_values.json"

        container_layout = QVBoxLayout()
        container_widget = QWidget()
        container_widget.setLayout(container_layout)
    
        form_layout = QFormLayout()

        browse_layout = QHBoxLayout()
        container_layout.addLayout(browse_layout)

        container_layout.addLayout(form_layout)

        browse_btn = QPushButton('Browse')
        browse_btn.clicked.connect(self.open_file_dialog)

        browse_label = QLabel("File: ")
        self.browse_line = QLineEdit()

        browse_layout.addWidget(browse_label)
        browse_layout.addWidget(self.browse_line)
        browse_layout.addWidget(browse_btn)


        # body_ant_ratio = 0.1
        # global_min_thresh = 120
        # top_weight = 1
        # bottom_weight = 1
        # left_weight = 1
        # right_weight = 1

        # mask_vals = (10, 20, 70, 40)

        update_vals_btn = QPushButton("Apply")
        update_vals_btn.clicked.connect(self.update_values_json)

        self.parameter_list = ["body ant ratio", "min thresh", "top weight", "bottom weight"
                          , "left weight", "right weight", "mask top", "mask bottom"
                          , "mask left", "mask right"]
        self.default_values = [0.1, 120, 1, 1, 1, 1, 10, 20, 70, 40]
        
        param_edit_list = []
        for param in self.parameter_list:
            param_edit = QLineEdit()
            # param_edit.setPlaceholderText(param)
            param_edit_list.append(param_edit)
            form_layout.addRow(param, param_edit)
        form_layout.addWidget(update_vals_btn)

        self.param_edit_list = param_edit_list
        self.setCentralWidget(container_widget)

    def open_file_dialog(self):
        file_filter = "Video Files (*.mp4 *.avi *.mov *.mkv)"
        filename, ok = QFileDialog.getOpenFileName(
            self,
            "Select a Raw Centipede File", 
            "", 
            file_filter
        )
        if filename:
            path = Path(filename)
            self.browse_line.setText(str(path))
            self.load_values_json()

    def load_values_json(self):
        filename = Path(self.browse_line.text()).name
        file = self.json_file
        with open(file, 'r') as json_file:
            try:
                json_data = json.load(json_file)
                if filename in json_data:
                    parameter_data = json_data[filename]
                    dict_values = list(parameter_data.values())
                    for count, edit in enumerate(self.param_edit_list):
                        edit.setText(str(dict_values[count]))
                else:
                    for count, edit in enumerate(self.param_edit_list):
                        default_value = str(self.default_values[count])
                        edit.setText(default_value)
            except json.JSONDecodeError:
                for count, edit in enumerate(self.param_edit_list):
                    default_value = str(self.default_values[count])
                    edit.setText(default_value)

    def update_values_json(self):
        file_path = self.browse_line.text()
        filename = Path(file_path).name
        param_values = []
        for idx, edit in enumerate(self.param_edit_list):
            text = edit.text()
            if text.strip() == "":
                QMessageBox.warning(self, "Input Error", f"Parameter '{self.parameter_list[idx]}' cannot be empty.")
                return
            param_values.append(text)
        
        parameter_data = dict(zip(self.parameter_list, param_values))
        file = self.json_file
        file_data = {}
        try:
            with open(file, 'r') as json_file:
                file_data = json.load(json_file)
        except:
            pass
        finally:
            file_data[filename] = parameter_data
            with open(file, 'w') as json_file:
                json.dump(file_data, json_file, indent=4)

        self.data = parameter_data
        self.filepath = file_path
        self.close()


set_directory_tree()
app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()

dict_values = window.data

min_x = float('inf')
max_x = 0
min_y = float('inf')
max_y = 0


body_ant_ratio = float(dict_values["body ant ratio"])
global_min_thresh = int(dict_values["min thresh"])
top_weight = int(dict_values["top weight"])
bottom_weight = int(dict_values["bottom weight"])
left_weight = int(dict_values["left weight"])
right_weight = int(dict_values["right weight"])

mask_vals = (int(dict_values["mask top"]), int(dict_values["mask bottom"]),
            int(dict_values["mask left"]), int(dict_values["mask right"]))

# default values
# body_ant_ratio = 0.1
# global_min_thresh = 120
# top_weight = 1
# bottom_weight = 1
# left_weight = 1
# right_weight = 1

# mask_vals = (10, 20, 70, 40)

endpoint1, endpoint2 = None, None
start_endpoint1,start_endpoint2 = None, None
endpoint1_dist = 0
endpoint2_dist = 0


filename = "polym_t4_d6.mp4"
filename = "subB_t3_d4.mp4"


full_path = window.filepath
filename = Path(full_path).name

def mask_frame(frame, mask_vals):
    top, bottom, left, right = mask_vals
    height, width = frame.shape
    frame[:top, :] = 0             # Top region
    frame[height - bottom:, :] = 0  # Bottom region
    frame[:, :left] = 0            # Left region
    frame[:, width - right:] = 0   # Right region
    return frame

def calc_vid_dims(height, width):
    tl = np.array((min_x, min_y)) #top left
    br = np.array((max_x, max_y)) # bottom right
    padding = np.linalg.norm(tl - br) * body_ant_ratio
    # body antennae ratio
    left = int(max(0, min_x - left_weight * padding))
    # left = int(max(0, min_x - 0.2*padding))
    right = int(min(width, max_x + right_weight * padding))
    # right = int(min(width, max_x + 0.2 * padding))
    top = int(max(0, min_y - top_weight * padding))
    # top = int(max(0, min_y - 0.6 * padding))
    bottom = int(min(height, max_y + bottom_weight * padding))
    # bottom = int(min(height, max_y + 0 * padding))
    return top, bottom, left, right

def crop_frame(frame, coords):
    top, bottom, left, right = coords
    new_frame = frame[top:bottom, left:right, :]
    return new_frame

def preprocess_frame(frame):
    # top, bottom, left, right
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray, (3,3), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))

    # adaptive thresholding to use different threshold 
    # values on different regions of the frame.
    blur = ~gray
    blur = mask_frame(blur, mask_vals)

    # bw = cv2.medianBlur(blur,9)

    ret, thresh = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    midline = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(midline)

  # Green rectangle
    new_min_x = min(min_x, x)
    new_max_x = max(max_x, x + w)
    new_min_y = min(min_y, y)
    new_max_y = max(max_y, y + h)

    # for contour in contours:
    #     x, y, w, h = cv2.boundingRect(contour)
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # return frame, thresh, (new_min_x, new_max_x, new_min_y, new_max_y)
    return (new_min_x, new_max_x, new_min_y, new_max_y)

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray, (3,3), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    midline_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # adaptive thresholding to use different threshold 
    # values on different regions of the frame.
    blur = ~gray

    # blur = cv2.medianBlur(blur,1)

    ret, thresh = cv2.threshold(blur, global_min_thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest_contour_index = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
    contours = [contour for i, contour in enumerate(contours) if i != largest_contour_index]
    cv2.drawContours(thresh, contours, -1, 0, thickness=cv2.FILLED)

    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, small_kernel)
    # thresh = cv2.medianBlur(thresh, 1)
    # eroded = cv2.erode(thresh, kernel, iterations=1)
    # dilated = 
    midline = cv2.erode(thresh, midline_kernel, iterations=1)


    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    opened = cv2.dilate(opened, small_kernel, iterations=1)

    # contours, _ = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cheeto = max(contours, key=cv2.contourArea)
    # x, y, w, h = cv2.boundingRect(cheeto)
    # print(x, y, w, h)

    # cheeto_mask = np.full(gray.shape, 0, dtype=np.uint8)
    # cv2.drawContours(cheeto_mask, cheeto, -1, (255), thickness=cv2.FILLED)
    legs_only_frame = thresh & ~opened

    # midline = midline > 0
    # midline = skeletonize(midline)
    # midline = (midline * 255).astype(np.uint8)
    midline = cv2.GaussianBlur(midline, (27, 27), 0)
    ret, midline = cv2.threshold(midline, 0, 255, cv2.THRESH_BINARY)
    binary_bool = midline > 0
    skeleton = skeletonize(binary_bool)
    skeleton = (skeleton * 255).astype(np.uint8)
    
    midline_contour, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    midline_contour = np.array(midline_contour)
    # print(midline_contour.shape)

    if midline_contour.shape[0] == 0:
        print("No midline contour found.")
    else:
        midline_contour = midline_contour[0, :, :, :]
        find_head(midline_contour)


    midline_and_legs = legs_only_frame | skeleton
    white_frame = ~midline_and_legs

    # binary_bool = midline_and_legs > 0
    # skeleton = skeletonize(binary_bool)

    # skeleton = (skeleton * 255).astype(np.uint8)
    # print(skeleton.shape)

    # skeleton = skeleton & ~midline
    # return skeleton

    return white_frame

def calculate_angle(A, B, C):
    # Convert points to NumPy arrays
    A, B, C = np.array(A), np.array(B), np.array(C)

    # Compute vectors BA and BC
    BA = A - B
    BC = C - B

    # Compute dot product and magnitudes
    dot_product = np.dot(BA, BC)
    mag_BA = np.linalg.norm(BA)
    mag_BC = np.linalg.norm(BC)

    # Avoid division by zero
    if mag_BA == 0 or mag_BC == 0:
        return 0  # Undefined angle

    # Compute angle in radians and convert to degrees
    cos_theta = np.clip(dot_product / (mag_BA * mag_BC), -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)

    # angle_rad = np.arctan2(np.linalg.norm(np.cross(BA, BC)), np.dot(BA, BC))
    # angle_deg = np.degrees(angle_rad)
    return angle_deg 

def find_head(mid_skele):
    global endpoint1, endpoint2, endpoint1_dist, endpoint2_dist, start_endpoint1, start_endpoint2
    epsilon = 0.003 * cv2.arcLength(mid_skele, True)
    midline_polygon = cv2.approxPolyDP(mid_skele, epsilon, True)

    polygon_squeezed = midline_polygon[:, 0, :]

    min1_angle = 360
    min2_angle = 360
    min_vertex = None
    min_vertex2 = None
    min_vertex_idx = 0
    min_vertex2_idx = 0

    for i in range(len(polygon_squeezed)): 
        # Get the current vertex and its two neighboring vertices
        pt1 = polygon_squeezed[i - 1]  # Previous point
        pt2 = polygon_squeezed[i]      # Current point
        pt3 = polygon_squeezed[(i + 1) % len(polygon_squeezed)]  # Next point
        
        # Calculate the angle at the current vertex (assuming calculate_angle is adapted to PyTorch)
        angle = calculate_angle(pt1.tolist(), pt2.tolist(), pt3.tolist())

        # Update minimum angle and associated vertex if a new minimum is found
        if angle < min1_angle:
            min1_angle, min2_angle = angle, min1_angle
            min_vertex2, min_vertex2_idx = min_vertex, min_vertex_idx
            min_vertex, min_vertex_idx = pt2, i
        elif angle < min2_angle:
            min2_angle = angle
            min_vertex2, min_vertex2_idx = pt2, i
        
    if endpoint1 is None or endpoint2 is None:
        endpoint1 = min_vertex
        endpoint2 = min_vertex2
        start_endpoint1 = endpoint1
        start_endpoint2 = endpoint2
    else:
        e1_m1= np.linalg.norm(endpoint1 - min_vertex)
        e1_m2= np.linalg.norm(endpoint1 - min_vertex2)
        e2_m1= np.linalg.norm(endpoint2 - min_vertex)
        e2_m2= np.linalg.norm(endpoint2 - min_vertex2)
        if e1_m1 < e1_m2:
            endpoint1_dist += e1_m1
            endpoint2_dist += e2_m2

            endpoint1 = min_vertex
            endpoint2 = min_vertex2
        else:
            endpoint1_dist += e1_m2
            endpoint2_dist += e2_m1

            endpoint1 = min_vertex2
            endpoint2 = min_vertex

def determine_head():
    if endpoint1_dist > endpoint2_dist:
        return endpoint1
    return endpoint2

def update_head_json(head):
    file = "head.json"
    data = {}
    try:
        with open(file, 'r') as json_file:
            data = json.load(json_file)
    except:
        pass
    finally:
        data[file_title] = head.tolist()
        with open(file, 'w') as json_file:
            json.dump(data, json_file, indent=4)


# get path to video and extract the file title
file_path = full_path
file_title = filename.split(".")[0]


cap = cv2.VideoCapture(file_path)

# ret, first_frame = cap.read()

# get video properties
if cap.isOpened():
    fps = cap.get(cv2.CAP_PROP_FPS)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    min_x, max_x, min_y, max_y = preprocess_frame(frame) # get the relevant video section only


cap.release()


cap_real = cv2.VideoCapture(file_path)



new_vid_coords = calc_vid_dims(height, width)
top, bottom, left, right = new_vid_coords
# print(new_vid_coords)
new_vid_width = right - left
new_vid_height = bottom - top
# print(new_vid_height, new_vid_width)

video_name = f"processed_videos/{file_title}_labelled.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_name, fourcc, fps, (new_vid_width, new_vid_height), isColor=False) 

while (cap_real.isOpened()):

    # Capture frame-by-frame
    # cv2.imshow("Processed", processed_frame)
    # cv2.imshow("color", color_frame)
    ret, frame = cap_real.read()
    if ret == False:
        break
    cropped_frame = crop_frame(frame, new_vid_coords)
    new_frame = process_frame(cropped_frame)
    cv2.imshow("Cropped Frame", new_frame)
    video.write(new_frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    # cv2.waitKey(0)

# release the vide
# o capture object
cap_real.release()
video.release()
# video.release()
# Closes all the windows currently opened.
cv2.destroyAllWindows()

head = determine_head()
update_head_json(head)



