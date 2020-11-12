from PIL import Image
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
from math import ceil
import numpy as np
import cv2


def prime_factors(number):
    factor = 2
    factors = []
    while factor * factor <= number:
        if number % factor:
            factor += 1
        else:
            number //= factor
            factors.append(int(factor))
    if number > 1:
        factors.append(int(number))
    return factors


def calculate_padding(kernel_size, stride=1, in_size=0):
    out_size = ceil(float(in_size) / float(stride))
    return int((out_size - 1) * stride + kernel_size - in_size)


def calculate_output_size(in_size, kernel_size, stride, padding):
    return int((in_size + padding - kernel_size) / stride) + 1


def is_power2(num):
    return num != 0 and ((num & (num - 1)) == 0)


def video_to_frames(video_filename, path):
    cap = cv2.VideoCapture(video_filename)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    frames = []
    if cap.isOpened() and video_length > 0:
        frame_ids = [0]
        if video_length >= 4:
            frame_ids = [0, 
                         round(video_length * 0.25), 
                         round(video_length * 0.5),
                         round(video_length * 0.75),
                         video_length - 1]
        count = 0
        success, image = cap.read()
        while success:
            if count in frame_ids:
                frames.append(image)
            success, image = cap.read()
            count += 1
    step = 1
    for pic in frames:
        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(pic)
        im.save(f"{path}/file{step}.bmp")
        step += 1
        
        
def extract_face_from_image(image_path, new_path, required_size=(200, 300)):
    image = plt.imread(image_path)
    detector = MTCNN()
    faces = detector.detect_faces(image)

    face_images = []

    for face in faces:
        x1, y1, width, height = face['box']
        x2, y2 = x1 + width, y1 + height

        face_boundary = image[y1:y2, x1:x2]

        face_image = Image.fromarray(face_boundary)
        face_image = face_image.resize(required_size)
        face_array = np.asarray(face_image)
        face_images.append(face_array)
        
        data = face_images[0]
        rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)

        im = Image.fromarray(rescaled)
        im.save(new_path)