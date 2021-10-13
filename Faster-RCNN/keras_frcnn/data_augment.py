import cv2
import os
import numpy as np
import copy
from tensorflow.keras import backend as K


def augment(img_data, config, labelName,augment=False):
	'''assert 'filepath' in img_data
	assert 'bboxes' in img_data
	assert 'width' in img_data
	assert 'height' in img_data

	img_data_aug = copy.deepcopy(img_data)'''
	images = []
	for i in img_data.index:
		try:
			for i in img_data.index:
				img = cv2.imread(os.path.join(img_data["Folder"][i], img_data["FileName"][i]))
			#img = cv2.imread(os.path.join(img_data["Folder"][img_data.index[0]],img_data["FileName"][img_data.index[0]]))
		except IndexError:
			print("Index error caught")
			print(img_data)

		if augment:
			rows, cols = img.shape[:2]

			if config.use_horizontal_flips and np.random.randint(0, 2) == 0:
				img = cv2.flip(img, 1)
				for bbox in img_data[labelName][i]:
					x1 = bbox['xmin']
					x2 = bbox['xmax']
					bbox['xmax'] = cols - x1
					bbox['xmin'] = cols - x2

			if config.use_vertical_flips and np.random.randint(0, 2) == 0:
				img = cv2.flip(img, 0)
				for bbox in img_data[labelName][i]:
					y1 = bbox['ymin']
					y2 = bbox['ymax']
					bbox['ymax'] = rows - y1
					bbox['ymin'] = rows - y2

			if config.rot_90:
				angle = np.random.choice([0,90,180,270],1)[0]
				if angle == 270:
					img = K.permute_dimensions(img, (1,0,2))
					img = cv2.flip(img, 0)
				elif angle == 180:
					img = cv2.flip(img, -1)
				elif angle == 90:
					img = K.permute_dimensions(img, (1,0,2))
					img = cv2.flip(img, 1)
				elif angle == 0:
					pass

				for bbox in img_data[labelName][i]:
					x1 = bbox['xmin']
					x2 = bbox['xmax']
					y1 = bbox['ymin']
					y2 = bbox['ymax']
					if angle == 270:
						bbox['xmin'] = y1
						bbox['xmax'] = y2
						bbox['ymin'] = cols - x2
						bbox['ymax'] = cols - x1
					elif angle == 180:
						bbox['xmax'] = cols - x1
						bbox['xmin'] = cols - x2
						bbox['ymax'] = rows - y1
						bbox['ymin'] = rows - y2
					elif angle == 90:
						bbox['xmin'] = rows - y2
						bbox['xmax'] = rows - y1
						bbox['ymin'] = x1
						bbox['ymax'] = x2
					elif angle == 0:
						pass

		img = img[:, :, (2, 1, 0)]  # BGR -> RGB
		img = img.astype(np.float32)
		img[:, :, 0] -= config.img_channel_mean[0]
		img[:, :, 1] -= config.img_channel_mean[1]
		img[:, :, 2] -= config.img_channel_mean[2]
		img /= config.img_scaling_factor

		images.append(img)
	return img_data, np.array(images)
