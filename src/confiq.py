# config.py
DATA_PATH = "data/mit-bih/"
WINDOW_SIZE = 187
NUM_CLASSES = 2  # Normal vs Abnormal
INPUT_SHAPE = (187, 1)
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001

CLASS_NAMES = {
    0: 'NORMAL',
    1: 'ABNORMAL'
}