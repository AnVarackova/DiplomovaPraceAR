import numpy as np
import cv2 as cv
import glob

images = glob.glob(r'imagesCalib/*.png')

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objectP = np.zeros((7*7, 3), np.float32)
objectP[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)
objectPoints = []
imagePoints = []

for frame in images:
    # Nacteni obrazu a prevedeni do cernobile
    img = cv.imread(frame)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Nalezeni sachovnice a jejich rohu v obraze
    ret, corners = cv.findChessboardCorners(gray, (7, 7), None)

    # Ulozeni bodu v obraze a vykresleni vnitrnich rohu sachovnice
    if ret:
        objectPoints.append(objectP)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imagePoints.append(corners)
        cv.drawChessboardCorners(img, (7, 7), corners2, ret)
        # cv.imshow('img', img)
        cv.imwrite('chess.jpg', img)
        cv.waitKey(500)

cv.destroyAllWindows()

# Ziskani kalibracnich parametru kamery
ret, cam_matrix, calib_coeffs, rvecs, tvecs = cv.calibrateCamera(objectPoints, imagePoints, gray.shape[::-1], None, None)

# Ukladani do souboru
np.save('calib_coeffs.npy', calib_coeffs)
np.save('cam_matrix.npy', cam_matrix)

print(cam_matrix)
print("rtvecs")
print(rvecs)
print(tvecs)
print("cal coef")
print(calib_coeffs)
print("ret")
print(ret)

