# jet-k-means
k-means and MoG (GMM) algorythms for anomalous jet tagging 

How to use:
1. The jet images used for this code were produced by the algorythm in a different repository please refere to it or create your own jet images using  "preprocessing.py".
1.1 Some plots require a mock dataset (5 dim gaussian distribution) these can be generated by running a script "mock_data.py"
2. image_data_sets_path.txt should contain the path to a directory with all jet image datasets. change it if needed.
3. dataset_path_and_pref.py takes a "DATASET" token and returns the dictionary with the information on the dataset namely the prefixes associated with it and its location. There are two prefixes one for background and one for signal and filepaths to background training set signal contamination set, and beackground and signal validataion (evaluation) sets. Optionally one can set edges for the list slise if both training and evaluation are parts of the same file. REVERSE=True changes places signal and background.
4. If the path to the datasets is given correctly script "full_run.py" will produce all the plots and a table given in k-menas chapter
5. k_means_process and MoG_process can be used with any other available parameters and datasets from dataset_path_and_pref.py at wish

The code is not fully commented and documented yet. It can also still be improved and simplified somewhat:
possible impreovements:
. Add function descriptions (Google style?)
. Change variable/function names for clarity and consistency
. refactor some code parts to make them more efficient logical
. deal with a huge number of input variables to many functions

code by:
Ivan Oleksiyuk

