This program theoretically works.

How to run:
1) First run `git clone https://github.com/dtan2537/centipede_tracker2.git` in your desired directory. Anytime something has been pushed, run `git pull` to update everything.
2) In a conda terminal, to create the necessary environment, run `conda env create -f environment.yml`
3) Everytime a script is to be run, you must be in the conda environment. You can do that with `conda activate $environment_name$`. If you forget the name, `conda env list` should list all available environments.
4) If you want to preprocess a video, run `python centi_preprocess.py`
   1) There will be UI adjust filter strength. Filters are listed in order of application.
   2) Adjust the parameters to achieve the clearest video possible (sample provided in processed_videos)
5) If you want to run the centipede tracker, change variable $full_file_path_to_video$ in centi_track.py to the preprocessed video path and run `python centi_track.py`
6) Plots and videos will be written to the `\out` directory

Basically how the program works:
1) Preprocess and extract initial head data
2) Find the midline and skeletonize it
3) Find all the legs and antennae
4) Run algorithm to get all shoulder/ feet
5) Using shoulders, traverse leg groups to find feet
6) Extract angles from legs
7) Plot