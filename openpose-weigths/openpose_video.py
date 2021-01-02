import numpy as np
from keras.layers import *
from keras.models import *
import matplotlib.pyplot as plt
import cv2 as cv
from tqdm import tqdm


def process_image(model, image_orig, detector, scale_mul=2, peaks_th=0.1, sigma=3):
    scale = 368/image_orig.shape[1]
    scale = scale*scale_mul
    image =  cv.resize(image_orig, (0,0), fx=scale, fy=scale)

    net_out = model.predict(np.expand_dims( image /256 -0.5 ,0))

    out = cv.resize( net_out[0], (image_orig.shape[1], image_orig.shape[0]) )
    image_out = image_orig

    mask = np.zeros_like(image_out).astype(np.float32)
    circles = []
    for chn in range(0, out.shape[-1]-2):
        m = np.repeat(out[:,:,chn:chn+1],3, axis=2)
        m = 255*( np.abs(m)>0.2)

        m_norm = cv.normalize(m, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8UC3)
        gray = cv.cvtColor(m_norm, cv.COLOR_BGR2GRAY)
        ret, gray = cv.threshold(gray,127,255,0)
        gray2 = gray.copy()
        mask1 = np.zeros(gray.shape,np.uint8)
        contours, hier = cv.findContours(gray, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if 200 < cv.contourArea(cnt) < 5000:
                cv.drawContours(m, [cnt], 0, (0,255,0), 2)
                cv.drawContours(mask1, [cnt], 0, 255, -1)
        print(contours, hier)
        plt.imshow(mask1)
        plt.show()

        plt.imshow(m)
        plt.show()

        mask = mask + m*(mask==0)

    mask = np.clip(mask, 0, 255)
    plt.imshow(mask)
    plt.show()
    image_out = image_out*0.8 + mask*0.2
    
    image_out = np.clip(image_out, 0, 255).astype(np.uint8)

    return image_out# , circles

model = load_model('model.h5')
libras_video = cv.VideoCapture('/home/lucas/Projects/Libras Video DB/db/Inventário+Libras/FLN G1 D1 CONVER Copa2014 v323/v0.mp4')

fourcc = int(cv.VideoWriter_fourcc(*'H264'))
video_width = int(libras_video.get(cv.CAP_PROP_FRAME_WIDTH))
video_height = int(libras_video.get(cv.CAP_PROP_FRAME_HEIGHT))
video_fps = int(libras_video.get(cv.CAP_PROP_FPS))
final_video = cv.VideoWriter('video_hand_poses.mp4', fourcc, video_fps,
                             (video_width, video_height))

params = cv.SimpleBlobDetector_Params()

params.filterByArea = True
params.minArea = 100

# Set Circularity filtering parameters
params.filterByCircularity = True
params.minCircularity = 0.9
# Set Convexity filtering parameters
params.filterByConvexity = True
params.minConvexity = 0.2

# Set inertia filtering parameters
params.filterByInertia = True
params.minInertiaRatio = 0.01
detector = cv.SimpleBlobDetector_create(params)

pbar = tqdm(total=video_fps * 10)
data = []
count = 0
while video_fps * 10:
    ret, frame = libras_video.read()
    image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    image = cv.resize(image, (video_width, video_width))
    image_out = process_image(model, image, detector)
    # data.append(circles)
    final_video.write(image_out)
    count += 1
    pbar.update(count)

pbar.close()
final_video.release()

# with open("skels.txt", "w") as txt_file:
#     for line in data:
#         txt_file.write(" ".join(line) + "\n")
