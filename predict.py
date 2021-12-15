# !/user/bin/env python    
# -*- coding:utf-8 -*-
# -----------------------------------------------------------------------#
#   What predict.py can do
#   1, output the prediction for a single image;
#   2, output real-time prediction using WebCam feed;
#   3, output the prediction for a whole directory;
#   4, test the number of real-time prediction fps on your computer using the image in "img/street.jpg"
#
#   change the variable mode to switch between different prediction tasks
# -----------------------------------------------------------------------#
import time
import os

import cv2
import numpy as np
from PIL import Image

# for HPC users, please uncomment the following line and change directory to project working directory
# os.chdir("/content/drive/MyDrive/YOLO_3")

from yolo import YOLO

if __name__ == "__main__":
	yolo = YOLO()
	# ----------------------------------------------------------------------------------------------------------#
	#   mode specifies the type of prediction tasks
	#   'predict' means single image prediction
	#   'video' means use WebCan feed to do real-time prediction
	#   'fps' means test the fps
	#   'dir_predict' means outputing the prediction for a whole directory
	# ----------------------------------------------------------------------------------------------------------#
	mode = "predict"
	# ----------------------------------------------------------------------------------------------------------#
	#   video_path specifies the video you wanna do prediciton on. Being 0 indicates using WebCam
	# ----------------------------------------------------------------------------------------------------------#
	video_path = 0
	video_save_path = ""
	video_fps = 25.0
	# -------------------------------------------------------------------------#
	#   test_interval specifies the number of tests we run using the test image when doing fps test
	# -------------------------------------------------------------------------#
	test_interval = 100
	# -------------------------------------------------------------------------#
	#   dir_origin_path specifies the directory for directory prediction
	#   dir_save_path specifies the save path for prediction results
	# -------------------------------------------------------------------------#
	dir_origin_path = "/content/drive/My Drive/YOLO_3/resource/demo_video/images"
	dir_save_path = "/content/drive/My Drive/YOLO_3/resource/demo_video/results"

	if mode == "predict":
		while True:
			img = input('Input image filename:')
			try:
				image = Image.open(img)
			except:
				print('Open Error! Try again!')
				continue
			else:
				r_image = yolo.detect_image(image)
				r_image.save("test_result.png")
				r_image.show()

	elif mode == "video":
		capture = cv2.VideoCapture(video_path)
		if video_save_path != "":
			fourcc = cv2.VideoWriter_fourcc(*'XVID')
			size = (
			int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
			out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

		ref, frame = capture.read()
		if not ref:
			raise ValueError("Cannot obtain WebCam")

		fps = 0.0
		while True:
			t1 = time.time()
			# read a frame
			ref, frame = capture.read()
			if not ref:
				break
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			frame = Image.fromarray(np.uint8(frame))
			frame = np.array(yolo.detect_image(frame))
			frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

			fps = (fps + (1. / (time.time() - t1))) / 2
			print("fps= %.2f" % (fps))
			frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
								(0, 255, 0), 2)

			cv2.imshow("video", frame)
			c = cv2.waitKey(1) & 0xff
			if video_save_path != "":
				out.write(frame)

			if c == 27:
				capture.release()
				break

		print("Video Detection Done!")
		capture.release()
		if video_save_path != "":
			print("Save processed video to the path :" + video_save_path)
			out.release()
		cv2.destroyAllWindows()

	elif mode == "fps":
		img = Image.open('img/street.jpg')
		tact_time = yolo.get_FPS(img, test_interval)
		print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')

	elif mode == "dir_predict":
		import os

		from tqdm import tqdm

		img_names = os.listdir(dir_origin_path)
		for img_name in tqdm(img_names):
			if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm',
										  '.ppm', '.tif', '.tiff')):
				image_path = os.path.join(dir_origin_path, img_name)
				image = Image.open(image_path)
				r_image = yolo.detect_image(image)
				if not os.path.exists(dir_save_path):
					os.makedirs(dir_save_path)
				r_image.save(os.path.join(dir_save_path, img_name))

	else:
		raise AssertionError(
			"Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
