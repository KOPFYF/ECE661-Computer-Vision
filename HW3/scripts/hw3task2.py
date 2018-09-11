import numpy as np
import cv2
import math
import os
import argparse

img1 = cv2.imread('../HW3Pics/1.jpg')
img2 = cv2.imread('../HW3Pics/2.jpg')
img1_coord = cv2.imread('../HW3WorldCoords/1.coords.jpg')
img2_coord = cv2.imread('../HW3WorldCoords/2.coords.jpg')


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


def Homography_affine(P,Q,S,R):
    '''
    Finding Homography using two orthogonal lines in the world plane
    H: 3 by 3 Homography matrix
    H_p: Homography that removes projective
    P: points coord
    C*x = B

    '''
    # P = np.append(P.T,1)

    # get othogonal lines
    l1 = np.cross(P, Q)
    m1 = np.cross(Q, S)
    l2 = np.cross(R, P)
    m2 = np.cross(P, Q)

    # 
    C = np.array([[l1[0]*m1[0], l1[0]*m1[1] + l1[1]*m1[0]], [l2[0]*m2[0], l2[0]*m2[1] + l2[1]*m2[0]]])
    # print('C:', C)
    B = np.array([[-l1[1]*m1[1]],[-l2[1]*m2[1]]])
    # print('B:', B)
    x = np.dot(np.linalg.inv(C), B)
    # print('x:', x)
    S = np.array([[x[0][0], x[1][0]],[x[1][0], 1]])
    # print('S:', S)
    u, d, v = np.linalg.svd(S)
    # print('u:', u)
    # print('d:', d)
    # print('v:', v)

    A = np.dot(np.dot(v,np.sqrt(d)),v.T)

    H = np.eye(3)
    H[0][0:2] = A[0]
    H[1][0:2] = A[1]
    print(H)

    return H



def Homography_vanish(P,Q,S,R):
    '''
    Finding Homography using two orthogonal lines in the world plane
    H: 3 by 3 Homography matrix
    H: Homography H_v
    P: points coordinates (x_i,y_i)
    l1 = P x Q
    l2 = Q x S
    l3 = S x R
    l4 = R x P

    '''
    H = np.eye(3)

    l1 = np.cross(P, Q)
    l2 = np.cross(Q, S)
    l3 = np.cross(S, R)  
    l4 = np.cross(R, P)

    vp1 = np.cross(l1,l3)
    vp2 = np.cross(l2,l4)
    vl = np.cross(vp1,vp2)
    print('vanish line0:', vl)
    # rescale
    vl = vl/np.max(vl)
    print('vanish line:', vl)
    H[2][:] = vl[:]
    print('H_vanish', H)

    return H


def load_point_for_two_step(args):
    if args.img_name == '1':
        # world coordinate
        P_w, Q_w, S_w, R_w = np.array([0,0,1]), np.array([60,0,1]), np.array([0,80,1]), np.array([60,80,1])
        X_w = [P_w, Q_w, S_w, R_w]
        # point to point 
        P_i, Q_i, S_i, R_i = np.array([1011,1857]), np.array([957,2064]), np.array([1269,1863]), np.array([1218,2076])
        X_i = [P_i, Q_i, S_i, R_i]
        # affine
        P_i2, Q_i2, S_i2, R_i2 = np.array([120,45,1]), np.array([245,45,1]), np.array([139,129,1]), np.array([242,138,1])
        X_i2 = [P_i2, Q_i2, S_i2, R_i2]
        # vl
        P_v, Q_v, S_v, R_v = np.array([1,1,1]), np.array([1,1,1]), np.array([1,1,1]), np.array([1,1,1])

        if args.two_step_method == 'p2p':
            H_p = Homography(X_w, X_i)
        elif args.two_step_method == 'vl':
            H_p = Homography_vanish(P_v, Q_v, S_v, R_v)
        else:
            raise Exception('Error: No such two step method!')

    elif args.img_name == '2':
        # world coordinate
        P_w, Q_w, S_w, R_w = np.array([0,0,1]), np.array([60,0,1]), np.array([0,80,1]), np.array([60,80,1])
        X_w = [P_w, Q_w, S_w, R_w]
        # point to point 
        P_i, Q_i, S_i, R_i = np.array([58,236,1]), np.array([283,237,1]), np.array([275,333,1]), np.array([73,335,1])
        X_i = [P_i, Q_i, S_i, R_i]
        # affine
        P_i2, Q_i2, S_i2, R_i2 = np.array([120,45,1]), np.array([245,45,1]), np.array([242,138,1]), np.array([139,129,1])
        X_i2 = [P_i2, Q_i2, S_i2, R_i2]
        # vl
        P_v, Q_v, S_v, R_v = np.array([58,236,1]), np.array([283,237,1]), np.array([275,333,1]), np.array([73,335,1])

        if args.two_step_method == 'p2p':
            H_p = Homography(X_w, X_i)
        elif args.two_step_method == 'vl':
            H_p = Homography_vanish(P_v, Q_v, S_v, R_v)
        else:
            raise Exception('Error: No such two step method!')
    elif args.img_name == '3':
        pass
    elif args.img_name == '4':
        pass
    else:
        raise Exception('Error: No such image!')

    return H_p, X_i, X_i2


def two_step(img, args):
    H_p, X_i, X_i2 = load_point_for_two_step(args)

    H_p_inv = np.linalg.inv(H_p)

    box_i = np.array([[1,1,1],[img.shape[1],1,1],[img.shape[0],1,1],[img.shape[1],img.shape[0],1]])
    # print(box_i.shape) # 4 by 3
    box_h = np.zeros((4,3))
    for i in range(box_i.shape[0]):
        box_h[i] = np.dot(H_p,box_i[i])
        box_h[i] = box_h[i]//box_h[i][2]
    print('box_h:', box_h)

    xymin_h = np.min(box_h, axis=0)
    xmin_h, ymin_h = xymin_h[0], xymin_h[1] 

    # removing affine: x_h = H_p*x-xmin
    X_0 = []
    for pt in X_i2:
        tmp = np.dot(H_p,pt) # 3 by 1
        print()
        pt_ = np.array([tmp[0][0]//tmp[2][0]-xmin_h, tmp[1][0]//tmp[2][0]-ymin_h])
        X_0.append(pt_)
    H_a = Homography_affine(X_0)
    H_a_inv = np.linalg.inv(H_a)

    H = H_a_inv * H_p
    # H = H_p
    H_inv = np.linalg.inv(H)
    X_i = np.array(X_i) # 4 by 3
    W = np.dot(H, X_i.T) # 3 by 3 times 3 by 4 is ---- 3 by 4

    xymin = np.min(W, axis=0)
    # print(xymin)
    xmin, ymin = xymin[0], xymin[1] 
    xymax = np.max(W, axis=0)
    xmax, ymax = xymax[0], xymax[1] 
    width_ = xmax - xmin
    height_ = ymax - ymin
    print('w & h', width, height)

    # write output
    img_out = np.zeros((height_, width_, 3))
    for h in range(height_):
        for w in range(width_):
            # scale back
            tmp = np.array([[w+xmin-1],[h+ymin-1],[1]]) # 3 x 1
            print(tmp)
            H_b = np.dot(H_inv, tmp) # 3 x 1
            x_i, y_i = H_b[0][0]//H_b[2][0], H_b[1][0]//H_b[2][0]
            print(x_i, y_i)
            img_out[h][w] = img[x_i][y_i]
    cv2.imwrite('../HW3Pics/twostep1.jpg',img_out)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_name', type=str, default=None) # 1， 2， 3， 4
    parser.add_argument('--two_step_method', type=str, default='p2p') # p2p or vl
    args = parser.parse_args()
    two_step(img1, args)



if __name__ == '__main__':
    main()
    # print()

