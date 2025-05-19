This program theoretically works.

How to run:
1) Create a venv/ conda environment with requirements.txt
2) If you want to preprocess a video, run `python centi_preprocess.py`
   1) There will be UI to help save previously used parameters
   2) Adjust the parameters to achieve the clearest video possible (sample provided in processed_videos)
3) If you want to run the centipede tracker, change variable "full_file_path_to_video" in centi_track.py and run `python centi_track.py`

Basically how the program works:
1) Preprocess and extract initial head data
2) Find the largest contour(midline) and segment it
3) Find all the legs and antennae
4) Skeletonize and run algorithm to get all shoulder/ feet
5) Using shoulders, traverse leg groups to find feet
6) Extract angles from legs
7) Plot