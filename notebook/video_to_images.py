mypath = '/Users/freddie/Downloads/video_split/normal'

import os
filenames = next(os.walk(mypath), (None, None, []))[2]  # [] if no file
filenames.sort()
print(filenames)

import cv2
for file in filenames:
    print('Processing video: ', file)
    try:
        os.mkdir(mypath + '/' + "_".join(file.split('_')[:-1]))
    except OSError as error:
        print(error)

    vidcap = cv2.VideoCapture(mypath + '/' + file)
    success, image = vidcap.read()
    frame_counter = 0
    counter = 0
    while success:
        if counter == 0:
            write_dir = mypath + '/' + "_".join(file.split('_')[:-1]) + "/" + "_".join(file.split('_')[:-1]) + "_frame_%d.jpg" % frame_counter
            print(write_dir)
            cv2.imwrite(write_dir, image)     # save frame as JPEG file
            frame_counter += 1

        success,image = vidcap.read()
        counter = (counter + 1) % 1
