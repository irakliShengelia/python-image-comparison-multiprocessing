import cv2
import numpy as np
import urllib.request
import pandas as pd
import time
import os
from multiprocessing import Process, current_process

rawDirectory = 'raw'
parsedDirectory = 'analyzed'

def url_to_image(url):
    	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
	resp = urllib.request.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
	# return the image
	return image

def compare(url1, url2):
	keypoints = 1000
	img1 = url_to_image(url1)
	img2 = url_to_image(url2)

	orb = cv2.ORB_create(nfeatures=keypoints)

	##Keypoints and descriptors
	kp1, des1 = orb.detectAndCompute(img1,None)
	kp2, des2 = orb.detectAndCompute(img2,None)

	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1,des2,k=2)

	good = []
	for m,n in matches:
		if m.distance < 0.5*n.distance:
			good.append([m])

	##Key Point Match Ratio
	return len(good)/keypoints
	##Generate Image for reference
	#img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)

def parseFile(file,filename):
	df = pd.read_csv(file)
	totalRows = len(df.index)
	start = time.time()
	for index, row in df.iterrows():
		try:
			probability = compare(row['imageUrl iubar.1'],row['Amazon B1.1'])
			df.at[index,'probability'] = probability
			timeElapsed = round(time.time() - start,2)
			timeRemaining = round((timeElapsed/(index+1)*(totalRows-index-1))/60,2)
			print('Parsing ' + str(index + 1) + '/' + str(totalRows) + '. Time Remaining: ' + str(timeRemaining) + 'minutes.', end='\r')
		except:
			print('Something went wrong', end='\r',)
	df.to_csv(parsedDirectory + '/' + filename, index = False)
	os.remove(file)


if __name__ == '__main__':
	processes = []

	for filename in os.listdir(rawDirectory):
		if filename.endswith('.csv'):
			process = Process(target=parseFile, args=(rawDirectory + '/'+filename, filename))
			# parseFile(rawDirectory + '/'+filename, filename)
			processes.append(process)

			process.start()