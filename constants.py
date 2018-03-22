# Resources
PREDICTOR_PATH = "shared/shape_predictor_68_face_landmarks.dat"

# Constants
SCALE_FACTOR = 0.4
SKIP_FRAME_VAL = 10
FEATHER_AMOUNT = 25
# SCREEN = [900,1440] #rows,cols (height, width)
# SCREEN = [768,1024] #rows,cols (height, width) PROJECTOR IN HORIZONTAL ORIENTATION
SCREEN = [1280, 720]  # PROJECTOR IN VERTICAL ORIENTATION

# Facial Landmarks based on ibug facial point annotations
landmark_ids = list(map(str, range(1, 69)))

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Image masking points
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS, JAW_POINTS + MOUTH_POINTS, FACE_POINTS
]
FOREHEAD_POINTS = [
    LEFT_BROW_POINTS + RIGHT_BROW_POINTS
]
