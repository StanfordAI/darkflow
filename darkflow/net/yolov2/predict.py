import numpy as np
import math
import cv2
import os
import json
#from scipy.special import expit
#from utils.box import BoundBox, box_iou, prob_compare
#from utils.box import prob_compare2, box_intersection
from ...utils.box import BoundBox
from ...cython_utils.cy_yolo2_findboxes import box_constructor
from PIL import Image
import numpy as np
#import cv2 as cv
face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haar/haarcascade_eye.xml')
 
def crop(image_path, coords, saved_location):
    """
    @param image_path: The path to the image to edit
    @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
    @param saved_location: Path to save the cropped image
    """
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop(coords)
    cropped_image.save(saved_location)
    #cropped_image.show()
    #return cropped_image
 
 
#if __name__ == '__main__':
#    image = './data/frame0.jpg'
#    crop(image, (161, 166, 706, 1050), 'cropped.jpg')


def expit(x):
	return 1. / (1. + np.exp(-x))

def _softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def findboxes(self, net_out):
	# meta
	meta = self.meta
	boxes = list()
	boxes=box_constructor(meta,net_out)
	return boxes

def postprocess(self, net_out, im, save = True):
	"""
	Takes net output, draw net_out, save to disk
	"""
	#print("postprocessing ... cool"+str(im))
	boxes = self.findboxes(net_out)

	# meta
	meta = self.meta
	threshold = meta['thresh']
	colors = meta['colors']
	labels = meta['labels']
	if type(im) is not np.ndarray:
		imgcv = cv2.imread(im)
	else: imgcv = im
	h, w, _ = imgcv.shape
	
	resultsForJSON = []
	b_idx = 0
	for b in boxes:
		boxResults = self.process_box(b, h, w, threshold)
		if boxResults is None:
			continue
		left, right, top, bot, mess, max_indx, confidence = boxResults
		#print("box index: " + str(b_idx))
		#print("destinatio:"+str(im)+"person_"+str(b_idx)+".jpg")
		
		original_file_path = str(im)
		people_folder = original_file_path+"_people"
		if not os.path.exists(people_folder):
			os.mkdir(people_folder)
		person_file_path = people_folder + "/person_" + str(b_idx) + "_at_x1_" + str(left) + "_y1_" + str(top) + "_x2_" + str(right) + "_y2_" + str(bot) + ".png"
		crop(original_file_path, (left, top, right, bot), person_file_path)
		#cv.imshow("Object found", object_image)
	
		
		person_img = cv2.imread(person_file_path)
		#cv.imshow("Person found", person_img)
		person_gray = cv2.cvtColor(person_img, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(person_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
		print("Found {0} faces!".format(len(faces)))
		f_idx = 0
		print("identify faces ..." + str(faces))
		for (x,y,w,h) in faces:

			faces_folder = person_file_path+"_faces"
			if not os.path.exists(faces_folder):
				os.mkdir(faces_folder)

			print("Processing a face " + str(f_idx) + " at " + person_file_path)
			face_file_path = faces_folder + "/face_" + str(f_idx) + "_at_x1_" + str(x) + "_y1_" + str(y) + "_x2_" + str(x+w) + "_y2_" + str(y+h) + ".png"
			print("Generating " + face_file_path)
			crop(person_file_path, (x,y,x+w,y+h), face_file_path)
			f_idx = f_idx + 1
		



		b_idx = b_idx + 1
		thick = int((h + w) // 300)
		if self.FLAGS.json:
			resultsForJSON.append({"label": mess, "confidence": float('%.2f' % confidence), "topleft": {"x": left, "y": top}, "bottomright": {"x": right, "y": bot}})
			continue

		cv2.rectangle(imgcv,
			(left, top), (right, bot),
			colors[max_indx], thick)
		cv2.putText(imgcv, mess, (left, top - 12),
			0, 1e-3 * h, colors[max_indx],thick//3)

	if not save: return imgcv

	outfolder = os.path.join(self.FLAGS.imgdir, 'out')
	img_name = os.path.join(outfolder, os.path.basename(im))
	if self.FLAGS.json:
		textJSON = json.dumps(resultsForJSON)
		textFile = os.path.splitext(img_name)[0] + ".json"
		with open(textFile, 'w') as f:
			f.write(textJSON)
		return

	cv2.imwrite(img_name, imgcv)
