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
    """Set up required directory structure for the application.
    """
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
    """UI for the main window of the application.
    """
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


        update_vals_btn = QPushButton("Apply")
        update_vals_btn.clicked.connect(self.update_values_json)

        self.parameter_list = ["body ant ratio", "min thresh", "top weight", "bottom weight"
                          , "left weight", "right weight", "mask top", "mask bottom"
                          , "mask left", "mask right"]
        self.default_values = [0.1, 120, 1, 1, 1, 1, 10, 20, 70, 40]
        
        param_edit_list = []
        for param in self.parameter_list:
            param_edit = QLineEdit()
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


class ProcessFrame():

    def __init__(self, user_params):
        self.body_ant_ratio = float(user_params["body ant ratio"])
        self.global_min_thresh = int(user_params["min thresh"])
        self.top_weight = float(user_params["top weight"])
        self.bottom_weight = float(user_params["bottom weight"])
        self.left_weight = float(user_params["left weight"])
        self.right_weight = float(user_params["right weight"])
        self.mask_vals = (int(user_params["mask top"]), int(user_params["mask bottom"]),
                            int(user_params["mask left"]), int(user_params["mask right"]))
        self.br = [0, 0]
        self.tl = [1000000, 1000000]
        self.body_end1 = None
        self.body_end2 = None
        self.first_frame_bodyend1 = None
        self.first_frame_bodyend2 = None
        self.body_end1_dist = 0
        self.body_end2_dist = 0

        self.full_path = window.filepath
        self.filename = Path(self.full_path).name
        self.file_title = self.filename.split(".")[0]
        self.mask_poly = None


    def poly_mask_frame(self, frame, val=0):
        """Mask components from window of video

        Args:
            frame (_type_): frame to be masked
            mask_vals (_type_): how much is masked from each side of the frame

        Returns:
            _type_: new frame
        """
        polygon = np.array([self.mask_poly], dtype=np.int32)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8) * 255
        cv2.fillPoly(mask, polygon, 255)
        masked = cv2.bitwise_and(frame, frame, mask=mask)
        return masked


    def pad_mask_frame(self, frame, preproc=True):
        """crop the frame to the relevant section
        """
        if len(frame.shape) > 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        top, bottom, left, right = self.mask_vals
        height, width = frame.shape[:2]
        mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
        mask[:top, :] = 0            # Top region
        mask[height - bottom:, :] = 0  # Bottom region
        mask[:, :left] = 0        # Left region
        mask[:, width - right:] = 0   # Right region
        if preproc:
            frame[mask==0] = 0
        else:
            frame[mask==0] = 255
        masked = frame

        if not preproc:
            tl_x, tl_y = self.tl
            br_x, br_y = self.br
            masked = masked[tl_y:br_y, tl_x:br_x]
        return masked

    def update_win_size(self):
        padding = int(self.body_length * self.body_ant_ratio)
        tl_x = max(self.tl[0] - int(self.left_weight * padding), 0)
        tl_y = max(self.tl[1] - int(self.top_weight * padding), 0)
        br_x = min(self.br[0] + int(self.right_weight * padding),width)
        br_y = min(self.br[1] + int(self.bottom_weight * padding), height)
        self.tl = (tl_x, tl_y)
        self.br = (br_x, br_y)

    def preprocess_frame(self, frame):
        """Preprocess the frame to find the midline of the centipede and find the relevant video section."""
        # top, bottom, left, right
        gray = frame
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))

        # adaptive thresholding to use different threshold 
        # values on different regions of the frame.
        inverted = ~gray
        masked = self.pad_mask_frame(inverted)
        
        cv2.imshow("masked", masked)

        ret, thresh = cv2.threshold(masked, 180, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        midline = max(contours, key=lambda x: cv2.arcLength(x, True))

        x, y, w, h = cv2.boundingRect(midline)
        self.body_length = int(cv2.arcLength(midline, True)/2)

        self.tl = (min(x, self.tl[0]), min(y, self.tl[1]))
        self.br = (max(x + w, self.br[0]), max(y + h, self.br[1]))
  

    def process_frame(self, frame):
        """Process the frame to leave only legs and midline of the centipede."""
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray = frame
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        midline_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # adaptive thresholding to use different threshold 
        # values on different regions of the frame.
        blur = ~gray


        ret, thresh = cv2.threshold(blur, self.global_min_thresh, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        largest_contour_index = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
        contours = [contour for i, contour in enumerate(contours) if i != largest_contour_index]
        cv2.drawContours(thresh, contours, -1, 0, thickness=cv2.FILLED)

        midline = cv2.erode(thresh, midline_kernel, iterations=1)


        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        opened = cv2.dilate(opened, small_kernel, iterations=1)
        legs_only_frame = thresh & ~opened


        midline = cv2.GaussianBlur(midline, (27, 27), 0)
        binary_bool = midline > 0
        skeleton = skeletonize(binary_bool)
        skeleton = (skeleton * 255).astype(np.uint8)
        
        midline_contour, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(midline_contour) == 0:
            raise Exception("Parameters are not valid for this video. Please adjust the parameters in the UI.")
        else:
            midline_contour = midline_contour[0]
            self.find_head(midline_contour)


        midline_and_legs = legs_only_frame | skeleton
        white_frame = ~midline_and_legs

        return white_frame
    
    def embed_frame(self, frame, shape):
        """Embed the processed frame back into the original frame size."""
        embedded = np.ones(shape[:2], dtype=np.uint8) * 255
        embedded[self.tl[1]:self.br[1], self.tl[0]:self.br[0]] = frame
        return embedded

    def calculate_angle(self, A, B, C):
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

        return angle_deg 

    def find_head(self, mid_skele):
        """Track the head of the centipede by determinging the vertex that is more exploratory"""
        epsilon = 0.003 * cv2.arcLength(mid_skele, True)
        midline_polygon = cv2.approxPolyDP(mid_skele, epsilon, True)
        midline_polygon += np.array(self.tl) #adjust for offset from optimization
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
            angle = self.calculate_angle(pt1.tolist(), pt2.tolist(), pt3.tolist())

            # Update minimum angle and associated vertex if a new minimum is found
            if angle < min1_angle:
                min1_angle, min2_angle = angle, min1_angle
                min_vertex2, min_vertex2_idx = min_vertex, min_vertex_idx
                min_vertex, min_vertex_idx = pt2, i
            elif angle < min2_angle:
                min2_angle = angle
                min_vertex2, min_vertex2_idx = pt2, i
            
        if self.body_end1 is None or self.body_end2 is None:
            self.body_end1 = min_vertex
            self.body_end2 = min_vertex2
            self.first_frame_bodyend1 = self.body_end1
            self.first_frame_bodyend2 = self.body_end2
        else:
            e1_m1= np.linalg.norm(self.body_end1 - min_vertex)
            e1_m2= np.linalg.norm(self.body_end1 - min_vertex2)
            e2_m1= np.linalg.norm(self.body_end2 - min_vertex)
            e2_m2= np.linalg.norm(self.body_end2 - min_vertex2)
            if e1_m1 < e1_m2:
                self.body_end1_dist += e1_m1
                self.body_end2_dist += e2_m2

                self.body_end1 = min_vertex
                self.body_end2 = min_vertex2
            else:
                self.body_end1_dist += e1_m2
                self.body_end2_dist += e2_m1

                self.body_end1 = min_vertex2
                self.body_end2 = min_vertex

    def determine_head(self):
        #TODO
        if self.body_end1_dist > self.body_end2_dist:
            return self.body_end1
        return self.body_end2

    def update_head_json(self, head):
        file = "head.json"
        data = {}
        try:
            with open(file, 'r') as json_file:
                data = json.load(json_file)
        except:
            pass
        finally:
            data[self.file_title] = head.tolist()
            with open(file, 'w') as json_file:
                json.dump(data, json_file, indent=4)



if __name__ == "__main__":
    set_directory_tree()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()

    dict_values = window.data

    proc_frame = ProcessFrame(dict_values)

    cap = cv2.VideoCapture(proc_frame.full_path)
    # get video properties
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    print("Preprocessing video...")
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        proc_frame.preprocess_frame(frame) # get the relevant video section only
        cv2.waitKey(25)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # reset video to first frame
    proc_frame.update_win_size()

    video_name = f"processed_videos/{proc_frame.file_title}_labelled.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height), isColor=False) 
    print("Processing video...")
    while (cap.isOpened()):

        # Capture frame-by-frame

        ret, frame = cap.read()
        if ret == False:
            break
        masked_frame = proc_frame.pad_mask_frame(frame, preproc=False)
        new_frame = proc_frame.process_frame(masked_frame)
        embedded_frame = proc_frame.embed_frame(new_frame, frame.shape)
        cv2.imshow("Processed", embedded_frame)

        video.write(embedded_frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # release the vide
    # o capture object
    cap.release()
    video.release()
    # Closes all the windows currently opened.
    cv2.destroyAllWindows()

    head = proc_frame.determine_head()
    proc_frame.update_head_json(head)

    print("Finished Processing")