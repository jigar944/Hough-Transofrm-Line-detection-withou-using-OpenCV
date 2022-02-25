
import cv2
import numpy as np

def hough_line_detection(img):
    width,height = img.shape
    theta = np.deg2rad(np.arange(0,180,1))
    len_theta = len(theta)
    diagonal_len = round(np.sqrt(width**2 + height**2))
    accumulator = np.zeros((2 * diagonal_len, len_theta), dtype=np.uint8)
    cos = np.cos(theta)
    sin = np.sin(theta)
    
    edge_pixels = np.where(img == 255)
    coordinates = list(zip(edge_pixels[0], edge_pixels[1]))

    for i in range(len(coordinates)):
        for j in range(len(theta)):
            rho = int(round(coordinates[i][1] * cos[j] + coordinates[i][0] * sin[j]))
            accumulator[rho, j] += 2

    return accumulator

if __name__ == '__main__':

    image = cv2.imread('hough1.png',cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(image,20,100)

    Hough_space = hough_line_detection(edges)
    
    edge_pixels = np.where(Hough_space > 110)
    coordinates = list(zip(edge_pixels[0], edge_pixels[1]))

   
    for i in range(0, len(coordinates)):
        t1 = np.cos(np.deg2rad(coordinates[i][1]))
        t2 = np.sin(np.deg2rad(coordinates[i][1]))
        x0 = t1*coordinates[i][0]
        y0 = t2*coordinates[i][0]
        x1 = int(x0 + 1000*(-t2))
        y1 = int(y0 + 1000*(t1))
        x2 = int(x0 - 1000*(-t2))
        y2 = int(y0 - 1000*(t1))

        cv2.line(image,(x1,y1),(x2,y2),(0,255,0),1)

    cv2.imshow("Hough space", Hough_space)
    cv2.namedWindow("Final_image", cv2.WINDOW_NORMAL)
    cv2.imwrite("Final_image.jpg", image)
    cv2.waitKey(0)