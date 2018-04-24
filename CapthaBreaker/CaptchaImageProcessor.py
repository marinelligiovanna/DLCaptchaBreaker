import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

class CaptchaImageProcessor:

    def hasBlackBackgroud(self, img):

        """
         Define if the thresholded image has a black or white backgroud.
         Uses K-means method w/ one cluster to decide

        :param img: Image path
        :return: True if the image has black background or false if not.

        """
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        img = np.array(img).reshape((img.shape[0] * img.shape[1], 3))

        # Cluster the pixel intensities
        cluster = KMeans(n_clusters=1)
        cluster.fit(img)

        if cluster.cluster_centers_[0][0] < 100:
            return True
        else:
            return False

    def thresholdCaptchaImage(self, captchaImagePath):

        """
        Apply Otsu's thresholding in captcha image for better character identification. Also applies other transformations (Erode, Dilate and Closing) to improve the thresholding result.

        :param captchaImagePath: Path of image to be thresholded
        :return: Thresholded image.
        """

        # Read image and resize for a better result in thresholding
        img = cv.imread(captchaImagePath, 0)
        h, w = img.shape[:2]
        img = cv.resize(img, (0, 0), fx=5, fy=5)
        gray = cv.cvtColor(img, cv.COLOR_BAYER_BG2GRAY)
        gray = cv.copyMakeBorder(gray, 8, 8, 8, 8, cv.BORDER_REPLICATE)

        # Apply gaussian filter and Otsu's thresholding
        blur = cv.GaussianBlur(gray, (7, 7), 0)
        ret, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        # Erode, dilate and closing to remove noise
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv.erode(thresh, kernel, iterations=1)
        thresh = cv.dilate(thresh, kernel, iterations=1)
        thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=1)

        # Return img resized back to its original size
        thresh = cv.resize(thresh, (w, h), interpolation=cv.INTER_CUBIC)

        # Guarantes that the image has a white background (suited to ML algorithm)
        if hasBlackBackgroud(thresh):
            thresh = 255 - thresh

        return thresh

    def segmentCaptchaImage(self, captchaImagePath, minWidht, minHeight):

        """
        Segment a thresholded captcha image using contours method. Find the bounding rectangle of each contour and crop the image according.

        :param minWidht:
        :param minHeight:
        :return:
        """

        img = cv.imread(captchaImagePath)
        if img is None:
            return []

        if img.shape != None and len(img.shape) == 3:
            gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        else:
            gray = img

        gray = cv.copyMakeBorder(gray, 8, 8, 8, 8, cv.BORDER_REPLICATE)
        chars = []

        # Get thresholded image with a white backgroud and black letters
        thresh = self.thresholdCaptchaImage(captchaImagePath)

        # Find contours in negative image to avoid finding the image borders
        groupCnts = cv.findContours(255 - thresh.copy(), cv.RETR_TREE,
                                    cv.CHAIN_APPROX_SIMPLE)
        groupCnts = groupCnts[1]

        clone = np.dstack([thresh.copy()] * 3)
        rectanglesCoordinates = []

        # OpenCV find contours from bottom of the image, so we need to store objects based on theirs x
        # to preserve the order
        for (i, c) in enumerate(groupCnts):

            # Compute the bounding box of the contour and keep only rectangles in ROI
            (x, y, w, h) = cv.boundingRect(c)
            minRect = cv.minAreaRect(c)

            if w >= minWidht and h >= minHeight:
                cv.rectangle(clone, (x, y), (x + w, y + h), (255, 0, 0), 1)
                rectanglesCoordinates.append([x, y, w, h])

        if len(rectanglesCoordinates) <= 0:
            return []

        # Sort according x-axis coordinates.
        zippedList = list(zip(*rectanglesCoordinates))
        xList = list(zippedList[0])
        dic = dict(zip(xList, rectanglesCoordinates))
        xList.sort()

        for x in xList:
            [x, y, w, h] = dic[x]
            charImg = thresh[y - 1:y + h + 1, x - 1:x + w + 1]
            charImg = cv.resize(charImg, (20, 20), interpolation=cv.INTER_CUBIC)
            chars.append(np.expand_dims(cv.cvtColor(charImg, cv.COLOR_GRAY2RGB), axis=0))

        return chars

    def process(self, captchaImagePath):

        minWidht = 5
        minHeight = 5

        segmentedChars = self.segmentCaptchaImage(captchaImagePath, minWidht, minHeight)

        return segmentedChars





