import numpy as np
import cv2
import math
import os

img1 = cv2.imread('../PicsHw2/1.jpg')
img2 = cv2.imread('../PicsHw2/2.jpg')
img3 = cv2.imread('../PicsHw2/3.jpg')
imgJ = cv2.imread('../PicsHw2/Jackie.jpg')


# Homogeneous points obtained from gimp
Points_1a = np.array([[176,1504,1], [2248,1492,1], [728,2948,1], [2056,3000,1]])
Points_1b = np.array([[336,1308,1], [2016,1308,1], [616,3008,1], [1904,3024,1]])
Points_1c = np.array([[736,920,1], [2088,904,1], [384,2808,1], [2240,2848,1]])
Points_1d = np.array([[0,0,1], [722,0,1], [0,1280,1], [722,1280,1]])


def Homography(pt, pt_prime):
	# Finding Homography A of Ax=b
    A = np.zeros((8,8))
    b = np.zeros((1,8))

    for i in range(0, len(pt)):
        A[i * 2] = [pt[i][0], pt[i][1], pt[i][2], 0, 0, 0,\
        (-1 * pt[i][0] * pt_prime[i][0]), (-1 * pt[i][1] * pt_prime[i][0])]
        A[i * 2 + 1] = [0,0,0,pt[i][0],pt[i][1],pt[i][2],\
        (-1 * pt[i][0] * pt_prime[i][1]), (-1 * pt[i][1] * pt_prime[i][1])]
        b[0][i * 2] = pt_prime[i][0]
        b[0][i * 2 + 1] = pt_prime[i][1]

    h = np.matmul(np.linalg.pinv(A),b.T)
    homography = np.zeros((3,3))
    homography[0] = h[0:3,0]
    homography[1] = h[3:6,0]
    homography[2][0:2] = h[6:8,0]
    homography[2][2] = 1

    return homography


def projection(image_src,image_target,pt,Homography):
    # initialize pixel as black
    temp = np.zeros((image_src.shape[0],image_src.shape[1],3), dtype='uint8')
    pts = np.array([[pt[0][1], pt[0][0]], [pt[1][1],pt[1][0]],\
    [pt[3][1], pt[3][0]], [pt[2][1], pt[2][0]]])
    # make the target area white
    cv2.fillPoly(temp,[pts],(255,255,255))

    for i in range(0,(image_src.shape[0]-1)):
        for j in range(0,(image_src.shape[1]-1)):
            if temp[i,j,1] == 255 and temp[i,j,0] == 255 and \
            temp[i,j,2]==255:
                point_tmp = np.array([i, j, 1])
                trans_coord = np.array(np.dot(Homography,point_tmp))
                # rescale then
                trans_coord = trans_coord/trans_coord[2]
                if (trans_coord[0] > 0) and (trans_coord[0] < image_target.shape[0]) and \
                (trans_coord[1] > 0) and (trans_coord[1] < image_target.shape[1]):
                    image_src[i][j] = image_target[math.floor(trans_coord[0]),math.floor(trans_coord[1])]
                    # temp[i][j] = image_target[math.floor(trans_coord[0]),math.floor(trans_coord[1])]
            else:
                continue

    return image_src


def projection2(image_src,image_target,pt,Homography):
    temp = np.zeros((image_src.shape[0],image_src.shape[1],3), dtype='uint8')

    for i in range(0,(image_src.shape[0]-1)):
        for j in range(0,(image_src.shape[1]-1)):
            point_tmp = np.array([i, j, 1])
            trans_coord = np.array(np.dot(Homography,point_tmp))
            trans_coord = trans_coord/trans_coord[2]
            if (trans_coord[0] > 0) and (trans_coord[0] < image_target.shape[0]) and \
            (trans_coord[1] > 0) and (trans_coord[1] < image_target.shape[1]):
                temp[i][j] = image_target[math.floor(trans_coord[0]),math.floor(trans_coord[1])]
    
    return temp


def write_img(pt1,pt2,output_file,input_img,imgJ):
	# Transforming the image in 1d to 1abc
	H = Homography(pt1, pt2)
	output = projection(input_img, imgJ, pt1, H)
	cv2.imwrite(output_file, output)


def write_img_a2c():
	# Transforming the image from a to c
	H_ab = Homography(Points_1b, Points_1a)
	H_bc = Homography(Points_1c, Points_1b)
	H_ac = np.matmul(H_ab, H_bc)

	Points_13 = np.array([[0,0],[0, img3.shape[0]],[img3.shape[1],0],[img3.shape[1],img3.shape[0]]])
	output13 = projection2(img3,img1,Points_13,H_ac)
	cv2.imwrite('../PicsHw2/image13.jpg',output13)


def main():
	write_img(Points_1a,Points_1d,'../PicsHw2/image1.jpg',img1,imgJ)
	write_img(Points_1b,Points_1d,'../PicsHw2/image2.jpg',img2,imgJ)
	write_img(Points_1c,Points_1d,'../PicsHw2/image3.jpg',img3,imgJ)
	write_img_a2c()


if __name__ == '__main__':
    main()

