
from multiprocessing.shared_memory import SharedMemory
import numpy as np
import easyocr
reader = easyocr.Reader(['en'])
import time


def ocr_detect(tmp, tmp2, width, height, rectangles_sum, contours_size):
    frame_memory = SharedMemory('FrameMemory')
    contours_memory = SharedMemory('ContoursMemory')

    while True:
        if contours_size.value != 0:
            frame_data = np.ndarray((height,width,3), dtype=np.uint8, buffer=frame_memory.buf)
            contours = reader.detect(frame_data, min_size=5, text_threshold=0.7)[0][0]

            if len(contours) > 0:
                # rectangles_sum.value = len(contours)
                # contours_data = np.ndarray((rectangles_sum.value, 4), dtype=np.int32, buffer=contours_memory.buf)
                contours_data = np.ndarray((len(contours), 4), dtype=np.int32, buffer=contours_memory.buf)
                contours_data[:, :] = contours
                rectangles_sum.value = len(contours)

            tmp.value = 1
            tmp2.value = 1

# def my_process(tmp, width, height, len_contours, contours_size):
 #   frame_memory = SharedMemory('FrameMemory')
 #   contours_memory = SharedMemory('ContoursMemory')

 #   while True:
 #       if contours_size.value != 0:
 #           frame_data = np.ndarray((height,width,3), dtype=np.uint8, buffer=frame_memory.buf)
 #           contours = reader.detect(frame_data, min_size=5, text_threshold=0.7)[0][0]
            # len_contours.value = len(contours) # перенести ниже

            # if len_contours.value > 0:
 #           if len(contours) > 0:
 #               len_contours.value = len(contours)
 #               contours_data = np.ndarray((len_contours.value, 4), dtype=np.int32, buffer=contours_memory.buf)
 #               contours_data[:, :] = contours
            # else:
                # contours_data = np.ndarray((1, 4), dtype=np.int32, buffer=contours_memory.buf)
                # contours_data[:, :] = [[0,0,0,0]]
                # len_contours.value = 1

#            tmp.value = 1


# def my_process(tmp, width, height, len_contours, contours_size):
#     frame_memory = SharedMemory('FrameMemory')
#     contours_memory = SharedMemory('ContoursMemory')
#
#     while True:
#         if contours_size.value != 0:
#             frame_data = np.ndarray((height,width,3), dtype=np.uint8, buffer=frame_memory.buf)
#             contours = reader.detect(frame_data, min_size=5, text_threshold=0.7)[0][0]
#             len_contours.value = len(contours)
#             contours_data = np.ndarray((len_contours.value, 4), dtype=np.int32, buffer=contours_memory.buf)
#             contours_data[:, :] = contours
#             tmp.value = 1



















