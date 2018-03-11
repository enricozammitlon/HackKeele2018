import sys
import os
import numpy as np
from PIL import Image
import cv2
from numpy.linalg import eig

with Image.open('helen.png') as helen:

	HELEN_ARRAY = np.array(helen.convert('L'),np.float)
	HELEN_ARRAY = cv2.resize(HELEN_ARRAY, (64,64))
	HELEN_ARRAY = HELEN_ARRAY.flatten()


images = []

location = "lfwcrop_grey/faces/"

for filename in os.listdir(location):
		im = location + filename
		images.append(im)#im.convert('RGB'))

#with Image.open(images[0]) as pic:
#	pic.show()

arr = np.zeros((64,64,1),np.float).flatten()

# Build up average pixel intensities, casting each image as an array of floats
#for image in images:
for im in images:
	with Image.open(im) as pic:
		imarr = np.array(pic.convert('L'),dtype=np.float).flatten()
		arr = arr+imarr/len(images)
##
# Round values in array and cast as 16-bit integer
mu = np.array(np.round(arr),dtype=np.uint16)

C = np.zeros((1,64*64,64*64), np.float)


A = []	

for im in images[0:1000]:
	with Image.open(im) as pic:
		print("MU:",mu.shape)
		phi = np.array(pic.convert('L'),dtype=np.float).flatten() - mu
		print(phi.shape)
		A.append(phi)
#		C += np.dot(phi, np.transpose(phi))

A = np.array(A)
#C = C/len(images)
C = np.dot(A, np.transpose(A))

print(A.shape)
print(C.shape)
print("Calculating eigenvalues:")
e_val, e_vec = eig(C)
print(e_vec.shape)

vecs = np.dot(np.transpose(A), e_vec)
print(vecs.shape)
vecs = np.transpose(vecs)

for row in vecs:
	print(row.shape)
#average_image = images[:,:,
print(vecs.shape)
out = Image.fromarray(vecs[980].reshape(64,64),mode="L")
if sys.argv[1]:
	imageName = sys.argv[1]

	faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

	image = cv2.imread(imageName)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
	faces = faceCascade.detectMultiScale(gray)
	for (x, y, w, h) in faces:
		crop = image[y:y+h, x:x+h]

	img = cv2.resize(crop, (64, 64))

	img_array = img.flatten

	vec = vecs.mean(axis=1)
	count = 0
	print(HELEN_ARRAY.shape)
	if np.linalg.norm(HELEN_ARRAY) > np.linalg.norm(img_array):
		while np.linalg.norm(HELEN_ARRAY) > np.linalg.norm(img_array):
			img_array += vec
			count += 1
	else:
		while np.linalg.norm(HELEN_ARRAY[0]) < np.linalg.norm(img_array[0]):
			img_array -= vec
			count += 1
	percentage = 1 - np.linalg.norm(count*vec)/np.linalg.norm(HELEN_ARRAY[0])
	while percentage > 1:
		percentage -= 1

print(string(percentage*1000))
out.show()
