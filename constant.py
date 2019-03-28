"""
The file is used to define constant parameters.
"""
import os


#######################################
# Preprocessing
AWS_PATH = "/extend/AWSdata"
SAMPLE_NUM = 100000
ZERO_TOLERANCE = 0
# Valid area of a ref radar echo map which constraint by the radar coverage.
VALID_LONGITUDE = [200, 500]        # Valid longitude
VALID_LATITUDE = [300, 600]         # Valid latitude

HEIGHT = 700
WIDTH = 900

ECHO_AREA = 61                      # The area of corresponding radar echo map

BASE_DIR = "/extend/rain_data"

NAME = "resize_31"

SAVE_DIR = os.path.join(BASE_DIR, "Save")
RESULT_DIR = os.path.join(BASE_DIR, "Results", NAME)
SUMMARY_DIR = os.path.join(SAVE_DIR, "Summary", NAME)
MODEL_DIR = os.path.join(SAVE_DIR, "Model", NAME)

DATA_DIR = os.path.join(BASE_DIR, "Data")
X_DIR = os.path.join(DATA_DIR, "X.npy")
Y_DIR = os.path.join(DATA_DIR, "Y.npy")

H1_DIR = os.path.join(DATA_DIR, "1H")
H1_DIR_F = os.path.join(DATA_DIR, "1H_F")
H1_DIR_LOG = os.path.join(DATA_DIR, "1H_LOG")
H1_RESAMPLE = os.path.join(DATA_DIR, "1H_Resample")

#######################################
# CNN config
FRAME_SIZE = ECHO_AREA
CNN_LEARNING_RATE = 0.001
CNN_BATCH_SIZE = 512
CNN_PREDICTION_SIZE = 1024

#######################################
ZR_a = 58.53
ZR_b = 1.56

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

