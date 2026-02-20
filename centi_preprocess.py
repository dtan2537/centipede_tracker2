import sys
import json
from pathlib import Path
import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage import io, img_as_float
from pathlib import Path
from dataclasses import dataclass
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
    QMessageBox,
    QSlider,
    QSpacerItem,
    QSizePolicy,
    QComboBox
)
from PyQt6 import QtCore
from PyQt6.QtGui import QImage, QPixmap


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
    Path("py_extensions").mkdir(parents=True, exist_ok=True)
    Path("process_values.json").touch(exist_ok=True)
    Path("head.json").touch(exist_ok=True)


class MainWindow(QMainWindow):
    """UI for the main window of the application.
    """
    def __init__(self):
        super().__init__()
        self.cap = None
        self.setWindowTitle("Centipede Preprocessing Parameters")
        self.data = None
        self.filepath = None
        self.param_panel = ParameterPanel()
        self.process_frame = ProcessFrame(self.param_panel.get_default_params())
        self.frame_panel = FramePanel(self.process_frame)
        self.add_on_python_extension = PythonExtension(self.process_frame)


        # self.resize(300, 200)

        container_layout = QVBoxLayout()
        container_widget = QWidget()
        container_widget.setLayout(container_layout)
    

        browse_layout = QHBoxLayout()
        container_layout.addLayout(browse_layout)

        panel_layout = QHBoxLayout()
        container_layout.addLayout(panel_layout)

        left_layout = QVBoxLayout()
    
        panel_layout.addLayout(left_layout)
        left_layout.addWidget(self.param_panel)


        panel_layout.addWidget(self.frame_panel)

        left_layout.addWidget(self.add_on_python_extension)

        browse_btn = QPushButton('Browse')
        browse_btn.clicked.connect(self.open_file_dialog)

        browse_label = QLabel("File: ")
        self.browse_line = QLineEdit()

        browse_layout.addWidget(browse_label)
        browse_layout.addWidget(self.browse_line)
        browse_layout.addWidget(browse_btn)

        
        update_vals_btn = QPushButton("Apply")
        update_vals_btn.clicked.connect(self.apply_btn_clicked)

        preproc_btn = QPushButton("Preprocess")
        preproc_btn.clicked.connect(self.preproc_btn_clicked)
        preproc_btn.setStyleSheet("""
            QPushButton {
                background-color: #b175ff;
                color: white;
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #8129f2; /* A slightly different purple on hover */
            }
            QPushButton:pressed {
                background-color: #4B0082; /* Indigo when clicked */
            }
        """)

        left_layout.addWidget(update_vals_btn)
        left_layout.addWidget(preproc_btn)

        spacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        left_layout.addItem(spacer)

        self.setCentralWidget(container_widget)

    def apply_btn_clicked(self):
        self.data = self.param_panel.get_user_params()
        self.frame_panel.set_parameter_dict(self.data)
        self.frame_panel.update_image_display()

    def preproc_btn_clicked(self):
        self.apply_btn_clicked()
        self.param_panel.update_values_json()
        self.close()

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
            self.filepath = path
            self.file_title = path.name.split(".")[0]
            self.browse_line.setText(str(path))

            if self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture(path)
            self.update_panels(path, self.cap)


    def update_panels(self, path, cap):
        self.param_panel.update(path=path, cap=cap)
        self.frame_panel.update(cap=cap)


class PythonExtension(QWidget):
    def __init__(self, process_frame):        
        super().__init__()
        self.layout = QHBoxLayout()
        self.filename = None
        self.mod_func = None
        self.process_frame = process_frame
        self.setLayout(self.layout)
        self.create_ui()

    def create_ui(self):
        self.label = QLabel("Python Add-on File")
        self.line_edit = QLineEdit()
        self.btn_btrowse = QPushButton("Browse")

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.line_edit)
        self.layout.addWidget(self.btn_btrowse)

        self.btn_btrowse.clicked.connect(self.open_file_dialog)
        self.line_edit.textChanged.connect(lambda text: setattr(self, 'filename', text))

    def open_file_dialog(self):
        # getOpenFileName returns a tuple (filename, selected_filter)
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select File",
            "", # Starting directory
            "Python Files (*.py)" # Filters
        )

        if filename:
            self.filename = filename
            self.line_edit.setText(filename)
            self.load_py(filename)

    def load_py(self, filename):
        #TODO: add error handling for invalid funcs
        import importlib.util
        spec = importlib.util.spec_from_file_location("user_module", filename)
        user_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(user_module)
        if hasattr(user_module, 'main'):
            self.mod_func = user_module.main
            self.process_frame.set_mod_func(self.mod_func)
        else:
            QMessageBox.warning(self, "Function Not Found", "The selected Python file does not have a 'main' function.")

    def py_extension_available(self):
        return self.mod_func is not None

@dataclass
class Parameter():
    name: str
    step_size: 1
    minimum: 0
    maximum: 100
    default: 0
    odd_only: bool = False
    ui_step_size: int = 1
    ui_minimum: int = 0
    ui_maximum: int = 100
    slider = None
    edit = None
    layout = None
    tooltip: str = ""
    

    def __post_init__(self):
        self.ui_step_size = 2 if self.odd_only else 1
        self.factor = self.ui_step_size / self.step_size
        self.ui_minimum = int(self.minimum * self.factor)
        self.ui_maximum = int(self.maximum * self.factor)
        self.value = self.default

    def create_ui(self):
        self.slider = QSlider()
        self.edit = QLineEdit()
        self.label = QLabel()
        self.layout = QHBoxLayout()
        self.layout.addWidget(self.slider)
        self.layout.addWidget(self.edit)
        self.set_slider()
        self.set_edit()
        self.set_label()

    def set_value(self, value):
        self.slider_func(float(value))
        self.edit_func()

    def get_value(self):
        value = None
        try:
            value = int(self.value)
        except Exception:
            value = float(self.value)
        return value

    def set_slider(self):
        self.slider.setFixedWidth(200)
        self.slider.setSingleStep(1)
        self.slider.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.slider.setMinimum(self.ui_minimum)
        self.slider.setMaximum(self.ui_maximum)
        self.slider.setValue(int(self.default * self.factor))

        self.slider.valueChanged.connect(self.slider_func)
        

    def set_edit(self):
        self.edit.setFixedWidth(50)
        self.edit.setText(str(self.default))
        self.edit.editingFinished.connect(self.edit_func)

    def set_label(self):
        self.label.setText(self.name)
        self.label.setToolTip(self.tooltip)

    def edit_func(self):
        text = self.edit.text()
        text_int = int(float(text) * self.factor) if text.replace('.','',1).isdigit() else 0
        text_int = max(self.ui_minimum, text_int - 1) if (self.odd_only and text_int % 2 == 0) else text_int
        # self.edit.setText(str(text_int))
        self.slider.setValue(text_int)

    def slider_func(self, slider_val):
        text_int = max(self.ui_minimum, slider_val - 1) if (self.odd_only and slider_val % 2 == 0) else slider_val
        self.edit.setText(f"{(text_int/self.factor):g}")
        self.value = self.edit.text()

    #TODO: add info bars

class ParameterPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.filepath = None
        self.json_file = "process_values.json"
        self.cap = None
        self.layout = QFormLayout()
        self.setLayout(self.layout)
        self.parameter_list = [Parameter("Global Threshold", step_size=1, minimum=0, maximum=255, default=160, tooltip="Set every pixel value abouve x to white"),
                               Parameter("Crop Top", step_size=1, minimum=0, maximum=100, default=0, tooltip="Mask out the top x pixels"),
                               Parameter("Crop Bottom", step_size=1, minimum=0, maximum=100, default=0, tooltip="Mask out the bottom x pixels"),
                               Parameter("Crop Left", step_size=1, minimum=0, maximum=100, default=0, tooltip="Mask out the left x pixels"),
                               Parameter("Crop Right", step_size=1, minimum=0, maximum=100, default=0, tooltip="Mask out the right x pixels"),
                               Parameter("Sharpen Amount", step_size=0.1, minimum=0, maximum=50, default=10.0, tooltip="Sharpen edges"),
                               Parameter("CLAHE Clip Limit", step_size=0.1, minimum=0, maximum=5.0, default=2.0, tooltip="Increase contrast variance"),
                               Parameter("Noise Filter Strength (H)", step_size=1, minimum=1, maximum=100, default=50, tooltip="Smooth out noisy patches"),
                               Parameter("Adaptive Thresh Size", step_size=2, minimum=3, maximum=201, default=101, odd_only=True, tooltip="Focus on larger objects"),
                               Parameter("Adaptive Thresh C", step_size=1, minimum=-100, maximum=100, default=50, tooltip="Clean background salt and pepper noise"),
                               Parameter("Midline Kernel Size", step_size=2, minimum=1, maximum=50, default=17, odd_only=True, tooltip="Increase range of centipede aware masking") ]
        for param in self.parameter_list:
            param.create_ui()
            self.layout.addRow(param.label, param.layout)

    def set_filepath(self, filepath):
        self.filepath = filepath

    def set_cap(self, cap):
        self.cap = cap

    def load_values_json(self):
        filename = self.filepath.name
        file = self.json_file
        with open(file, 'r') as json_file:
            try:
                json_data = json.load(json_file)
                if filename in json_data:
                    parameter_data = json_data[filename]
                    dict_values = list(parameter_data.values())
                    for i in range(len(self.parameter_list)):
                        param = self.parameter_list[i]
                        param.set_value(dict_values[i]) 
            except json.JSONDecodeError:
                pass

    def get_user_params(self):
        parameter_data = {param.name: param.value for param in self.parameter_list}
        self.data = parameter_data
        return parameter_data
    
    def get_default_params(self):
        parameter_data = {param.name: param.default for param in self.parameter_list}
        return parameter_data
    
    def update(self, path, cap):
        self.set_filepath(path)
        self.set_cap(cap)
        image_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        image_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        dim_dict = {"Crop Top": image_height,
         "Crop Left": image_width,
         "Crop Right": image_width,
         "Crop Bottom": image_height}
        for param in self.parameter_list:
            if param.name in dim_dict:
                param.slider.setMaximum(dim_dict[param.name])
        self.load_values_json()

    def update_values_json(self):
        filename = self.filepath.name
        parameter_data = self.get_user_params()
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

        # self.filepath = file_path


class ImagePipeline(dict):
    steps = ["original", "grayscale", "modded", "clahe", "sharpened", "denoised", "adapted", "combined", "cleaned", "final"]
    def __init__(self):
        super().__init__()
        self.add_default_keys()
    
    def add_default_keys(self):
        for key in ImagePipeline.steps:
            self[key] = None
    


class FramePanel(QWidget):
    def __init__(self, process_frame):
        super().__init__()
        self.filepath = None
        self.cap = None
        self.process_frame = process_frame
        self.parameter_dict = self.process_frame.user_params

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.frame_form = QHBoxLayout()

        self.frame_option_label = QLabel("Pipeline Stage:")

        self.frame_option_dropdown = QComboBox()
        self.options = ImagePipeline.steps
        self.frame_option = self.options[-2]
        self.frame_option_dropdown.setCurrentIndex(6)
        
        for i in range(len(self.options)):
            self.frame_option_dropdown.addItem(self.options[i], userData=i)

        self.frame_form.addWidget(self.frame_option_label)
        self.frame_form.addWidget(self.frame_option_dropdown)


        self.frame_label = QLabel("Frame: ")
        self.frame_form.addWidget(self.frame_label)

        self.frame_edit = QLineEdit()
        self.frame_edit.setFixedWidth(50)
        self.frame_form.addWidget(self.frame_edit)

        self.frame_slider = QSlider()
        self.frame_slider.setSingleStep(1)
        self.frame_slider.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(100)  # Placeholder maximum value

        self.frame_slider.valueChanged.connect(lambda value: self.frame_edit.setText(str(value)))

        self.frame_edit.textChanged.connect(lambda text: self.frame_slider.setValue(int(text) if text.isdigit() else 0))

        self.frame_slider.setValue(0)
        self.frame_edit.setText(str(self.frame_slider.value()))

        self.layout.addLayout(self.frame_form)
        self.frame_form.addWidget(self.frame_label)
        self.frame_form.addWidget(self.frame_slider)
        self.frame_form.addWidget(self.frame_edit)
        
        self.max_dim = 600
        self.placeholder_image = np.zeros((self.max_dim, self.max_dim), dtype=np.uint8)
        self.image_label = QLabel()
        self.update_image_display()
        self.layout.addWidget(self.image_label)
        

    def set_filepath(self, filepath):
        self.filepath = filepath

    def set_cap(self, cap):
        self.cap = cap

    def set_parameter_dict(self, parameter_dict):
        self.parameter_dict = parameter_dict

    def on_frame_slider_change(self, value):
        self.frame_edit.setText(str(value))
        self.update_image_display()

    def update_image_display(self):
        if self.cap is None:
            self.image = self.placeholder_image
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_slider.value())
            ret, frame = self.cap.read()
            if ret:
                self.process_frame.set_user_params(self.parameter_dict)
                index = self.frame_option_dropdown.currentData()
                images_dict = self.process_frame.process_frame(frame)
                image = list(images_dict.values())[index]
                # print(gray_frame.dtype)he he
                self.image = image
            else:
                self.image = self.placeholder_image
            cv2.waitKey(1)

        print(self.image.shape)
        image_height, image_width = self.image.shape[:2]

        if self.image.ndim == 2:
            image_format = QImage.Format.Format_Grayscale8
        else:
            image_format = QImage.Format.Format_BGR888
        
        qimage = QImage(self.image.data, self.image.shape[1], self.image.shape[0], self.image.strides[0], image_format)

        pixmap = QPixmap.fromImage(qimage.copy())
        scaled_pixmap = pixmap
        print(image_height, image_width, self.max_dim)
        if image_height > self.max_dim or image_width > self.max_dim:
            print("scaled to height")

            if image_height > image_width:
                scaled_pixmap = pixmap.scaledToHeight(self.max_dim, QtCore.Qt.TransformationMode.SmoothTransformation)
            else:
                scaled_pixmap = pixmap.scaledToWidth(self.max_dim, QtCore.Qt.TransformationMode.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

        if not self.process_frame.is_valid_frame:
            QMessageBox.warning(self, "Input Error", f"Parameters produce an invalid frame")
        # Update display logic would go here (e.g., set pixmap on a QLabel in the UI)
        # Example: self.image_label.setPixmap(pixmap)

    def update(self, cap):
        self.set_cap(cap)
        self.frame_slider.setMaximum(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
        self.update_image_display()

class ProcessFrame():

    def __init__(self, user_params):
        self.body_ant_ratio = 0.1
        self.top_weight = 1.0
        self.bottom_weight = 1.0
        self.left_weight = 1.0
        self.right_weight = 1.0
        self.global_min_thresh = int(user_params["Global Threshold"])
        self.mask_vals = (int(user_params["Crop Top"]), int(user_params["Crop Bottom"]),
                            int(user_params["Crop Left"]), int(user_params["Crop Right"]))
        self.br = [0, 0]
        self.tl = [1000000, 1000000]
        self.body_end1 = None
        self.body_end2 = None
        self.first_frame_bodyend1 = None
        self.first_frame_bodyend2 = None
        self.body_end1_dist = 0
        self.body_end2_dist = 0

        self.mask_poly = None

        self.backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        self.is_valid_frame = True
        self.mod_func = None
        
        self.user_params = user_params
        self.set_user_params(user_params)

    def set_user_params(self, user_params):
        self.global_min_thresh = int(user_params["Global Threshold"])
        self.mask_vals = (int(user_params["Crop Top"]), int(user_params["Crop Bottom"]),
                            int(user_params["Crop Left"]), int(user_params["Crop Right"]))
        self.sharpen = float(user_params["Sharpen Amount"])
        self.clahe_clip = float(user_params["CLAHE Clip Limit"])
        self.noise_strength = int(user_params["Noise Filter Strength (H)"])
        self.adapt_size = int(user_params["Adaptive Thresh Size"])
        self.adapt_c = int(user_params["Adaptive Thresh C"])
        self.midline_kernel_size = int(user_params["Midline Kernel Size"])

    def set_mod_func(self, func):
        self.mod_func = func

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
        
        cv2.imshow("Cropped Video", masked)

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


    def morph_reconstruction(self, marker, mask):
        """
        Regrows the 'marker' seeds until they fill the 'mask' boundaries.
        Used here to recover thin legs while ignoring isolated noise spots.
        """
        kernel = np.ones((3, 3), np.uint8)
        while True:
            expanded = cv2.dilate(marker, kernel)
            new_marker = cv2.bitwise_and(expanded, mask)
            if (new_marker == marker).all():
                break
            marker = new_marker
        return marker

    def unsharp_mask(self, image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
        """
        Return a sharpened version of the image using unsharp masking.
        """
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        sharpened = float(amount + 1) * image - float(amount) * blurred
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        if threshold > 0:
            low_contrast_mask = np.absolute(image - blurred) < threshold
            np.copyto(sharpened, image, where=low_contrast_mask)
            
        return sharpened

    def process_frame(self, frame):
        frame_save = frame.copy()
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(8,8))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray_save = gray.copy()

        mod = gray
        if self.mod_func is not None:
            mod = self.mod_func(gray)
            print(self.mod_func)

        mod_save = mod.copy()

        denoised = cv2.fastNlMeansDenoising(mod, None, h=self.noise_strength, templateWindowSize=7, searchWindowSize=21)
        denoise_save = denoised.copy()

        equalized = cv2.equalizeHist(denoised)

        cl = clahe.apply(equalized)
        clahe_save = cl.copy()
        result = self.unsharp_mask(cl, amount=self.sharpen, threshold=1)
        sharp_save = result.copy()
        kernel = np.ones((1,1), np.uint8)
        
        # mask = cv2.adaptiveThreshold(denoised, 255, 
        #                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        #                                 cv2.THRESH_BINARY, 101, 10)
        adaptive_thresh = cv2.adaptiveThreshold(sharp_save, 255, 
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, self.adapt_size, self.adapt_c)
        adaptive_save = adaptive_thresh.copy()
        cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel, iterations=10)

        cropped_gray = self.pad_mask_frame(~mod, preproc=True)
        ret, global_thresh = cv2.threshold(cropped_gray, self.global_min_thresh, 255, cv2.THRESH_BINARY)

        adapt_thresh = self.pad_mask_frame(~cleaned, preproc=True)

        thresh = adapt_thresh | global_thresh

        combined_save = thresh.copy()

        midline_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.midline_kernel_size, self.midline_kernel_size))

        midline = cv2.erode(thresh, midline_kernel, iterations=1)
        midline = cv2.dilate(midline, midline_kernel, iterations=2)
        comical_midline = cv2.dilate(midline, midline_kernel, iterations=8)

        thresh[comical_midline == 0] = 0

        # just for today
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel, iterations=1)

        cleaned_save = thresh.copy()

        white_frame = thresh.copy()
        try:
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            largest_contour_index = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
            thresh_filtered = np.zeros_like(thresh)
            cv2.drawContours(thresh_filtered, contours, largest_contour_index, 255, thickness=cv2.FILLED)
            thresh[thresh_filtered == 0] = 0

            midline_shape = cv2.GaussianBlur(midline, (27, 27), 0) > 0
            skeleton = skeletonize(midline_shape)
            skeleton = (skeleton * 255).astype(np.uint8)
            
            thresh[midline>0] = 0
            centi_processed = thresh | skeleton
            white_frame = ~centi_processed

            midline_contour, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            midline_contour = midline_contour[0]
            self.find_head(midline_contour)
            self.is_valid_frame = True

        except Exception:
            self.is_valid_frame = False

        images_dict = ImagePipeline()
        images_dict["original"] = frame_save
        images_dict["grayscale"] = gray_save
        images_dict["modded"] = mod_save
        images_dict["clahe"] = clahe_save
        images_dict["sharpened"] = sharp_save
        images_dict["denoised"] = denoise_save
        images_dict["adapted"] = adaptive_save
        images_dict["combined"] = combined_save
        images_dict["cleaned"] = cleaned_save
        images_dict["final"] = white_frame

        return images_dict
    
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

    def update_head_json(self, head, filepath):
        filename = Path(filepath).name
        file_title = filename.split(".")[0]
        json_filename = "head.json"
        data = {}
        try:
            with open(json_filename, 'r') as json_file:
                data = json.load(json_file)
        except:
            pass
        finally:
            data[file_title] = head.tolist()
            with open(json_filename, 'w') as json_file:
                json.dump(data, json_file, indent=4)



if __name__ == "__main__":
    set_directory_tree()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()

    dict_values = window.data

    proc_frame = window.process_frame

    cap = cv2.VideoCapture(window.filepath)
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

    video_name = f"processed_videos/{window.file_title}_processed.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height), isColor=False) 
    print("Processing video...")
    while (cap.isOpened()):

        # Capture frame-by-frame

        ret, frame = cap.read()
        if ret == False:
            break
        # gray_frame = proc_frame.gray_frame(frame)
        new_frame = proc_frame.process_frame(frame)["final"]
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
    proc_frame.update_head_json(head, window.file_title)

    print("Finished Processing")