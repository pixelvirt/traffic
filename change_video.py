import cv2
import numpy as np

import argparse
import os

# PURPOSE: Parsing the command line input and extracting the user entered values
def parseCommandLineArguments():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input", required=True,
		help="path to input video")
	ap.add_argument("-o", "--output", required=True,
		help="path to output video")
	args = vars(ap.parse_args())

	inputVideoPath = args["input"]
	outputVideoPath = args["output"]

	return inputVideoPath, outputVideoPath

inputvideo, outputvideo = parseCommandLineArguments()

cap = cv2.VideoCapture(inputvideo)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(outputvideo,fourcc, 5, (1280,720))

while True:
    ret, frame = cap.read()
    if ret == True:
        b = cv2.resize(frame,(1280,720),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
        out.write(b)
    else:
        break
    
cap.release()
out.release()
cv2.destroyAllWindows()