from utils import video_to_frames, extract_face_from_image
from sdfa import SDFA
import subprocess


command = "ffmpeg -i example/source1.avi -ab 160k -ac 2 -ar 44100 -vn example/audio.wav"
subprocess.call(command, shell=True)
video_to_frames('example/source2.avi', 'example/frames')
extract_face_from_image('example/frames/file4.bmp', 'example/face.bmp')

va = SDFA(gpu=0, model_path="data/model.dat")
vid, aud = va("example/face.bmp", "example/audio.wav")
va.save_video(vid, aud, "result/result_video.mp4")