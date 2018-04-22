import cv2
import numpy as np 
from os import listdir
from os.path import isfile,join
import glob
import capture

capture.capt()

images = [cv2.imread(file) for file in glob.glob('images/pos/*.jpg')]
count = 0


def change_gamma(image,gamma=1.0):
	invgamma = 1.0/gamma
	table = np.array([((i/255.0)** invgamma) * 255
		for i in np.arange(0,256)]).astype('uint8')
	return cv2.LUT(image,table)

def increase_gamma(image,count):
	gamma = 1.5
	changed  = change_gamma(image,gamma=gamma)
	cv2.imwrite('images\pos\increased_1.5g_{}.jpg'.format(count),changed)
	return 0

def decrease_gamma(image,count):
	gamma = 0.85
	changed  = change_gamma(image,gamma=gamma)
	cv2.imwrite('images\pos\decreased_0.75g_{}.jpg'.format(count),changed)
	return 0

def gen_img():
	count = 0
	for image in images:
			increase_gamma(image,count)
			decrease_gamma(image,count)	
			count +=1

if __name__ == '__main__':
	gen_img()
	


'''
for image in images:
	a = np.double(image)
	b = a + 30
	img2 = np.uint8(b)
	cv2.imwrite('images\increased_30_{}.jpg'.format(int(count_inc)),img2)
	count_inc+=1

for image in images:
	a = np.double(image)
	b = a + 60
	img2 = np.uint8(b)
	cv2.imwrite('images\increased_60_{}.jpg'.format(int(count_dec)),img2)
	count_dec+=1
'''