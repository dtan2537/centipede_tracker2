import cv2
import numpy as np
import numpy.typing as npt

def main(frame: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    # Invert the grayscale image
    background_subtracted = background_subtract(r"C:\Users\data\Documents\gatech assignments\CrabLab\centipede_tracker2\zzz.png", frame)
    return ~background_subtracted


def background_subtract(bg_img_path:str, frame):
    # bg_image = cv2.imread(bg_img_path)
    # if bg_image is None:
    #     print("Error: Could not load background image.")
    #     return

    # # Pre-process background: Convert to grayscale and blur to reduce noise
    # # bg_gray = cv2.cvtColor(bg_image, cv2.COLOR_BGR2GRAY)
    # # bg_gray = cv2.GaussianBlur(bg_gray, (21, 21), 0)

    # # # Ensure the frame is the same size as the background
    # # # (Needed if the image and video resolutions differ)
    # # frame_resized = cv2.resize(frame, (bg_image.shape[1], bg_image.shape[0]))
    
    # # # Pre-process current frame
    # # # gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    # # gray_frame = cv2.GaussianBlur(frame_resized, (21, 21), 0)

    # # # 3. Calculate Absolute Difference
    # # # This highlights the pixels that have changed
    # # frame_delta = cv2.absdiff(bg_gray, gray_frame)

    # # # 4. Thresholding
    # # # Convert the difference into a binary mask (Black/White)
    # # # If the difference is > 25, set it to 255 (white)
    # # threshold = 3
    # # threshold_frame = cv2.threshold(frame_delta, threshold, 255, cv2.THRESH_BINARY)[1]

    # # # 5. Dilate the thresholded image to fill in holes
    # # threshold_frame = cv2.dilate(threshold_frame, None, iterations=2)

    # # foreground_extracted = cv2.bitwise_and(frame, frame, mask=threshold_frame)
    # # return foreground_extracted

    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # enhanced = clahe.apply(frame)
    # blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    # edges = cv2.Canny(blurred, 30, 100)
    
    # # 2. Closing the Canny Lines
    # kernel = np.ones((9,9), np.uint8)
    # closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    # # 3. Watershed Setup (Distance Transform & Markers)
    # dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
    # _, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
    # sure_fg = np.uint8(sure_fg)
    
    # sure_bg = cv2.dilate(closing, kernel, iterations=3)
    # unknown = cv2.subtract(sure_bg, sure_fg)
    
    # _, markers = cv2.connectedComponents(sure_fg)
    # markers = markers + 1
    # markers[unknown == 255] = 0

    # # 4. Run Watershed
    # cv2.watershed(frame, markers)
    
    # # 5. Create the Final Binary Mask
    # # Labels > 1 are the "flooded" foreground areas
    # mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    # mask[markers > 1] = 255 

    # # --- WINDOW A: THE OVERLAY (Verification) ---
    # overlay = frame.copy()
    # overlay[mask == 255] = [0, 255, 0] # Green tint
    # final_overlay = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    # # Add white boundary line
    # final_overlay[markers == -1] = [255, 255, 255]

    # # --- WINDOW B: THE FOREGROUND (Extraction) ---
    # # We use bitwise_and to keep ONLY the pixels inside the mask
    # foreground_only = cv2.bitwise_and(frame, frame, mask=mask)
    # # return ~foreground_only 
    return frame




if __name__ == "__main__":
    main()