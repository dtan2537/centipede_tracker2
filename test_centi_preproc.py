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

        self.backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

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

    def gray_frame(self, frame):
        """Convert frame to grayscale if not already.
        """
        if len(frame.shape) > 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def pad_mask_frame(self, frame, preproc=True, white_bg=False):
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
        frame[mask==0] = 0 if not white_bg else 255
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
  

    def mask_top_wall(self, frame, left_pad, right_pad):
        """Mask the top wall of the centipede video based on the left and right padding values."""
        height, width = frame.shape[:2]
        polygon = np.array([[
            (0, 0),
            (width, 0),
            (0, left_pad),
            (width, right_pad)
        ]], dtype=np.int32)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8) * 255
        cv2.fillPoly(mask, polygon, 255)
        masked = cv2.bitwise_and(frame, frame, mask=mask)
        return masked

    def process_frame(self, frame):
        """Process the frame to leave only legs and midline of the centipede."""

        # kernels used for morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        midline_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        soft_midline_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        soft_midline_kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        circle_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        long_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 1))

        # use background subtraction to create a mask of just centipede outline and possible reflections
        fgMask = self.backSub.apply(frame)

        # convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        offset = 3
        top, bottom, left, right = self.mask_vals
        # gray = self.mask_top_wall(gray, left_pad=top - self.offset, right_pad= top + self.offset)
        cv2.imwrite("top_wall_masked.png", gray)

        # normalize and enhance contrast by removing uneven lighting
        norm_gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        blur = cv2.GaussianBlur(norm_gray, (3,3), 0)
        norm_gray = cv2.addWeighted(norm_gray, 1.5, blur, -0.5, 0)

        cv2.imwrite("1unevenlighting.png", norm_gray)

        # apply global thresholding to get just midline mask
        cropped_gray = self.pad_mask_frame(~gray, preproc=True)
        ret, global_thresh = cv2.threshold(cropped_gray, 170, 255, cv2.THRESH_BINARY)

        cv2.imwrite("2globalthresh.png", global_thresh)


        # apply histogram equalization and adaptive thresholding to get legs mask
        norm_gray = self.clahe.apply(norm_gray)
        adapt_thresh = cv2.adaptiveThreshold(norm_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,51,21)
        adapt_thresh = ~adapt_thresh

        cv2.imwrite("3norm_gray.png", norm_gray)
        cv2.imwrite("4adaptivethresh.png", adapt_thresh)


        # crop edges of
        adapt_thresh = self.pad_mask_frame(adapt_thresh, preproc=True)

        thresh = adapt_thresh | global_thresh

        cv2.imwrite("5combinedthresh.png", thresh)
        

        midline  = cv2.erode(global_thresh, small_kernel, iterations=2)
        midline = cv2.morphologyEx(midline, cv2.MORPH_CLOSE, small_kernel, iterations=1)
        midline = cv2.dilate(midline, midline_kernel, iterations=1)

        # midline = cv2.erode(midline, midline_kernel, iterations=1)

        # midline = cv2.erode(midline, soft_midline_kernel2, iterations=1)
        # midline = cv2.dilate(midline, soft_midline_kernel2, iterations=1)
        cv2.imwrite("5findmidline.png", midline)





        comical_midline = cv2.dilate(midline, midline_kernel, iterations=8)
        cv2.imwrite("comical_midline.png", comical_midline)

        # comical_midline[0,:] = 0
        midline = cv2.erode(comical_midline, midline_kernel, iterations=8)
        cv2.imshow("midline after comical", midline)


        thresh[comical_midline == 0] = 0
        thresh_copy = cv2.dilate(thresh, circle_kernel, iterations=3)
        cv2.imwrite("thresh.png", thresh_copy)
        

        # edge_tolerance = 15
        # edge_mask = np.zeros_like(thresh)
        # top, bottom, left, right = self.mask_vals
        # edge_mask[0:top+edge_tolerance, :] = 255
        # cv2.imwrite("edge_mask_top.png", edge_mask)
        # edge_section = np.zeros_like(thresh)
        # edge_section[edge_mask==255] = thresh[edge_mask==255]
        # # erode_section = cv2.morphologyEx(edge_section, cv2.MORPH_ERODE, long_kernel, iterations=1)



        # fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, circle_kernel, iterations=2)
        # erode_section = edge_section & fgMask
        # thresh[edge_mask==255] = erode_section[edge_mask==255]

        cv2.imwrite("6combineMOG.png", thresh)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # areas = np.array([cv2.contourArea(c) for c in contours])
        # second_max_index = np.argsort(areas)[:-2] if len(areas) > 1 else None

        largest_contour_index = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
        thresh_filtered = np.zeros_like(thresh)
        cv2.drawContours(thresh_filtered, contours, largest_contour_index, 255, thickness=cv2.FILLED)
        # thresh[thresh_filtered == 0] = 0

        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, midline_kernel)
        opened = cv2.dilate(opened, small_kernel, iterations=1)
        # legs_only_frame = thresh & ~opened
        # fgMask = self.pad_mask_frame(fgMask, preproc=True)

        legs_only_frame = thresh
        # legs_only_frame = (thresh | fgMask) 

        

        
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


        legs_only_frame = cv2.morphologyEx(legs_only_frame, cv2.MORPH_CLOSE, circle_kernel, iterations=1)
        # legs_only_frame = cv2.morphologyEx(legs_only_frame, cv2.MORPH_ERODE, circle_kernel, iterations=1)
        legs_only_frame = cv2.medianBlur(legs_only_frame, 7)


        # im_floodfill = legs_only_frame.copy()
        # h, w = legs_only_frame.shape[:2]
        # mask2 = np.zeros((h+2, w+2), np.uint8)
        # cv2.floodFill(im_floodfill, mask2, (0,0), 255)
        # im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        # legs_only_frame = legs_only_frame | im_floodfill_inv
        # legs_only_frame = cv2.GaussianBlur(legs_only_frame, (3, 3), 0)
        # legs_only_frame[legs_only_frame > 0] = 255
        # contours, _ = cv2.findContours(legs_only_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(legs_only_frame, contours, -1, 255, thickness=cv2.FILLED)

        

        legs_only_frame[binary_bool > 0] = 0


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
        cv2.waitKey(1)

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
        # gray_frame = proc_frame.gray_frame(frame)
        new_frame = proc_frame.process_frame(frame)
        # embedded_frame = proc_frame.embed_frame(new_frame, frame.shape)
        embedded_frame = new_frame
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