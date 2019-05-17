import numpy as np 
from numpy import fft
import math
from scipy import signal
from skimage import filters
from skimage import io, color, transform, img_as_ubyte, img_as_float
from matplotlib import pyplot
import random
from scipy import ndimage


#https://www.peterkovesi.com/papers/ai97.pdf
#https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1527851
mask = color.rgb2gray(img_as_float(io.imread("/Users/purvigoel/224Final/images/bunny_mask.png")))
image =  img_as_float(io.imread("/Users/purvigoel/224Final/images/bunny.png"))
image = color.rgb2gray(image)

for i in range(0, image.shape[0]):
	for j in range(0, image.shape[1]):
		if(mask[i,j] < 0.2):
			if(i % 2):
				image[i,j] = -1
			else:
				image[i,j] = -2
		
imageFFT = fft.fft2(image)

scales = 2 #4
orientations = 4 #6 

sigmaF = 0.35 #0.65
wavelengthIncrement = 3 #1.7
thetaStd = np.pi / orientations / 1.2

phaseSymmetry = np.zeros((image.shape[0], image.shape[1]))

symmetryTotal = np.zeros((image.shape[0], image.shape[1]))
amplitudeTotal = np.zeros((image.shape[0], image.shape[1]))
noiseAtOrientation = np.zeros((image.shape[0], image.shape[1]))
floor =  np.zeros((image.shape[0], image.shape[1]))

bank = []
for a in range(1,orientations + 1):
	centerAngle = (a - 1) * np.pi/orientations

	print(a)
	wavelength = 10 #3

	symmetryAtOrientation = np.zeros((image.shape[0], image.shape[1]))
	amplitudeAtOrientation = np.zeros((image.shape[0], image.shape[1]))
	kernels = []
	for n in range(scales):
		kernel = np.zeros((image.shape[0], image.shape[1]))
		centerFrequency = 1/wavelength
		for i in range(kernel.shape[0]):
			for j in range(kernel.shape[1]):
				y = i - (kernel.shape[0]/2)
				x = j-(kernel.shape[1]/2)

				normalizedY = y / (kernel.shape[0]/2)
				normalizedX = x / (kernel.shape[1]/2)

				normalizedRadius = math.sqrt(normalizedY * normalizedY + normalizedX * normalizedX)

				elementRadial = np.exp(-1 * np.power( (normalizedRadius/ (centerFrequency / 0.5)),2) / (2 * (sigmaF) * (sigmaF)) )
				theta = math.atan2(-y,x)
				deltaSin = math.sin(theta) * math.cos(centerAngle) - math.cos(theta) * math.sin(centerAngle)
				deltaCosine = math.cos(theta) * math.cos(centerAngle) + math.sin(theta) * math.sin(centerAngle)

				deltaTheta = abs(math.atan2(deltaSin, deltaCosine))

				elementAngular = np.exp((-1 * deltaTheta * deltaTheta) / (2 * thetaStd * thetaStd))

				kernel[i, j] = elementAngular * elementRadial
				if(kernel[i,j] < 0.001):
					kernel[i,j] = 0
		kernel = fft.fftshift(kernel)
		kernels.append(kernel)
		wavelength = wavelength * wavelengthIncrement
	
	bank.append(kernels)


def calculatePhaseSymmetry(imageFFT, phaseSymmetry, symmetryTotal, amplitudeTotal):
	for a in range(1,orientations + 1):
		symmetryAtOrientation = np.zeros((image.shape[0], image.shape[1]))
		amplitudeAtOrientation = np.zeros((image.shape[0], image.shape[1]))
		for n in range(scales):		
			kernel = bank[a - 1][n]		
			convolved = imageFFT * kernel
			s1 = np.array(kernel.shape)
			s2 = np.array(imageFFT.shape)
			convolved = fft.ifft2(convolved)
			
			evens = np.real(convolved)
			odds = np.imag(convolved)
			amplitude = np.sqrt(np.power(evens, 2) + np.power(odds, 2)) 
			
			amplitudeAtOrientation += amplitude
			symmetryAtOrientation += np.maximum((np.abs(evens) - np.abs(odds))  , floor)
		
		amplitudeTotal += np.add(amplitudeAtOrientation, 0.00001)
		symmetryTotal += symmetryAtOrientation

	phaseSymmetry = np.divide(symmetryTotal , amplitudeTotal)
	phaseSymmetry[mask < 0.2] = 0.0
	return phaseSymmetry

def proposeImageVector(mask, xmin, xmax, ymin, ymax, length):
	vector = []
	for i in range(vectorLength):
		lineLength = random.randint(5, 100)
		x = random.randint(xmin, xmax)
		y = random.randint(ymin, ymax)

		direction = random.uniform(-1 * math.pi, math.pi)		
		width = random.randint(1, 10)
		vector.append(x)
		vector.append(y)
		vector.append(lineLength)
		vector.append(direction)
		vector.append(width)
	return vector

def printImage(image, filename):
	saveImage = image.copy()
	saveImage[image < 0]= 0
	saveImage[image >1] = 1
	io.imsave(filename, img_as_ubyte(saveImage))

def vectorToImage(image, vector):
	new = image.copy()
	circles = np.zeros((image.shape[0], image.shape[1]))
	for i in range(0, len(vector), 5):
		x = vector[i]
		y = vector[i+1]
		lineLength = vector[i+2]
		angle = vector[i + 3]
		width = vector[i+4]
		taper = width
		tInc = width/lineLength
		if(abs(angle) <= math.pi/4 or abs(angle) > 3 * math.pi/4):
			for lx in range(lineLength):
				yCoordFloor = math.floor(lx * math.tan(angle))
				yCoordCeil = math.ceil(lx * math.tan(angle))
				taper = taper - tInc

				for w in range(math.ceil(taper)):
					if(abs(y + w + yCoordFloor) < image.shape[0] and x + lx < image.shape[1]):
						new[y + yCoordFloor + w][lx + x] = 0
					if(abs(y + w + yCoordCeil) < image.shape[0] and x + lx < image.shape[1]):
						new[y + yCoordCeil + w][lx + x] = 0
		else:
			for ly in range(lineLength):
				xCoordFloor = math.floor(ly / math.tan(angle))
				xCoordCeil = math.ceil(ly / math.tan(angle))
				taper = taper - tInc

				for w in range(math.ceil(taper)):
					if(abs(x + w + xCoordFloor) < image.shape[1] and y + ly < image.shape[0]):
						new[y + ly ][xCoordFloor + x + w] = 0
					if(abs(x + w + xCoordCeil) < image.shape[1] and y + ly < image.shape[0]):
						new[y + ly][xCoordCeil + x + w] = 0
	return new

def shiftOldVector(bestVector, xmin, xmax, ymin, ymax):
	vector = []

	for i in range(0, len(bestVector), 5):
		x = bestVector[i]
		y = bestVector[i+1]
		lineLength = bestVector[i+2]
		angle = bestVector[i+3]
		width = bestVector[i+4]

		r = random.randint(1, 5)
		if(r < 2): #6
			change = random.randint(-1, 1)
			x += change
			if x < xmin:
				x += xmin
			elif x > xmax:
				x -= xmax
		r = random.randint(1, 5)
		if(r < 2):#6
			change = random.randint(-1,1) 
			y += change
			if y < ymin:
				y += ymin
			elif y > ymax:
				y -= ymax

		r = random.randint(1, 5)
		if(r < 2):#6
			change = random.randint(-2,2)
			lineLength += change
			if lineLength <= 1:
				lineLength += abs(change)
			if lineLength > 100: #25 / 5
				lineLength -= abs(change)
		r = random.randint(1, 5)
		if(r < 2):#6
			change = random.uniform(-math.pi/8, math.pi/8)
			angle += change
			if angle < -1 * math.pi:
				angle += abs(change)
			if angle >  math.pi: #25 / 5
				angle -= abs(change)
		r = random.randint(1, 5)
		if(r < 2):#6
			change = random.randint(-1, 1)
			width += change
			if width <= 3:
				width += abs(change)
			if width >  10: #25 / 5
				width -= abs(change)
		vector.append(x)
		vector.append(y)
		vector.append(lineLength)
		vector.append(angle)
		vector.append(width)
	return vector

def maskMinMax(mask):
	minX = 10000
	minY = 100000
	maxX = 0
	maxY = 0
	for i in range(mask.shape[0]):
		for j in range(mask.shape[1]):
			if(mask[i,j] > 0.1):
				if i < minY:
					minY = i
				elif i > maxY:
					maxY = i

				if j < minX:
					minX = j 
				elif j > maxX:
					maxX = j 
	return minX, maxX, minY, maxY

def crop(image, xmin, xmax, ymin, ymax):
	new = np.zeros((image.shape[0], image.shape[1]))
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			if(i > ymin and i < ymax and j > xmin and j < xmax):
				new[i,j]= image[i, j]
	return new

desired = io.imread("/Users/purvigoel/224Final/images/bunny_desired.png")
desired = img_as_float(color.rgb2gray(desired))


# vector = []

# i = 0
# j = 0
# desired[desired < 0.2] = 0
# desired[desired > 0.2] = 1
# vector = []
# while i < image.shape[0]:
# 	startX = 0
# 	startY = i
	
# 	while startX < image.shape[1]:
# 		scanX = 0	
# 		while desired[i, startX] > 0.2:
# 			startX += 1
# 		scanX = startX
# 		while desired[i, scanX] < 0.2:
# 			scanX += 1

# 		length = scanX - startX

# 		vector.append(startX)
# 		vector.append(startY)
# 		vector.append(length)
# 		vector.append(0)
# 		vector.append(10)

# 		startX = scanX
# 	i += 10

# image = vectorToImage(image, vector)

# imageFFT = fft.fft2(image)
phaseSymmetry = calculatePhaseSymmetry(imageFFT, phaseSymmetry, symmetryTotal, amplitudeTotal)
io.imsave("/Users/purvigoel/224Final/images/phaseSymmetry.jpg", img_as_ubyte(phaseSymmetry))


image = (io.imread("/Users/purvigoel/224Final/images/bunny.png"))
image = color.rgb2gray(image)
image = img_as_float(image)

for i in range(0, image.shape[0]):
	for j in range(0, image.shape[1]):
		if(mask[i,j] < 0.2):
			if(i % 2):
				image[i,j] = -2
			else:
				image[i,j] = -1


bestDiff = 10000000
bestImage = np.zeros((image.shape[0], image.shape[1]))
bestCaustic = np.zeros((image.shape[0], image.shape[1]))

success = 0
attempt = 0

bestVector = []
printcounter = 0
xmin, xmax, ymin, ymax = maskMinMax(mask)
print(xmin, xmax, ymin, ymax)
eps = 1

while(success < 2000 and attempt < 200):
	r = random.randint(1, 10)
	attempt += 1
	vector = []
	if(r < 8 or len(bestVector) == 0):
		vectorLength = random.randint(100, 1000) #200, 1500 / 50,100
		vector = proposeImageVector(mask, xmin, xmax, ymin, ymax, vectorLength)
		proposal = vectorToImage(image, vector)
	else:

		vector = shiftOldVector(bestVector, xmin, xmax, ymin, ymax)
		proposal = vectorToImage(image, vector)


	candidate = proposal.copy()

	proposalFFT = fft.fft2(proposal)
	symmetryTotal = np.zeros((image.shape[0], image.shape[1]))
	amplitudeTotal = np.zeros((image.shape[0], image.shape[1]))
	phaseSymmetry = np.zeros((image.shape[0], image.shape[1]))
	caustic = calculatePhaseSymmetry(proposalFFT, phaseSymmetry, symmetryTotal, amplitudeTotal)		
	
	#distEdges = np.sum(np.abs(filters.sobel(caustic) - desiredEdges))
	# diff = np.sum(dist) + len(vector)
	diff = np.sum(np.abs(caustic - desired)) #+ len(vector)*0.1
	# diffOutsideDesiredCaustic = np.sum(diff[desired < 0.5])
	# diffInsideDesiredCaustic = np.sum(diff[desired > 0.5])
	# diff = diffOutsideDesiredCaustic * 2 + distEdges * 2 + diffInsideDesiredCaustic + len(vector)

	if attempt > 50:
		print("attempt", attempt)
	if attempt == 100 or attempt == 150 or attempt == 198:
		eps *= 1.5

	if(diff < bestDiff) or abs(bestDiff - diff) < eps:
		eps = 0.5
		success += 1
		print( diff, success)
		attempt = 0

		bestDiff = diff
		bestImage = candidate
		bestCaustic = caustic.copy()
		bestVector = vector	

	printcounter += 1
	if(printcounter == 100):
		print("saving images")
		printcounter = 0
		printImage(bestImage, "/Users/purvigoel/224Final/images/bestMap.jpg")
		printImage(bestCaustic, "/Users/purvigoel/224Final/images/bestCaustic.jpg")

printImage(bestImage, "/Users/purvigoel/224Final/images/bestMap.jpg")
printImage(bestCaustic, "/Users/purvigoel/224Final/images/bestCaustic.jpg")
