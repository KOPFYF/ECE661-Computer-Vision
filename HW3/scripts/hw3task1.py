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


def Homography_affine(P,Q,R,S):
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
    m1 = np.cross(Q, R)
    l2 = np.cross(R, P)
    m2 = np.cross(S, Q)

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
    # rescale
    vl = vl/np.max(vl)
    print('vanish line:', vl)
    H[2][:] = vl[:]
    print('H_vanish', H)

    return H


def Homography_1step(P0):
    '''
    input P0: 20 points in image forming 5 pairs of orthogonal lines in the world plane, 11 by 2
    P: 11 by 3
    output H: 3 by 3 Homography matrix
    C*x = B
    '''
    # print('P0',P0)
    P = np.zeros((P0.shape[0],3))
    for i in range(P0.shape[0]):
        P[i] = np.append(P0[i],1)
    # print(P)

    # find 5 pairs of orthogonal lines
    l_group = np.zeros((5,3))
    m_group = np.zeros((5,3))
    C = np.zeros((5,5))
    B = np.zeros((5,1))

    for i in range(l_group.shape[0]):
        l_group[i] = np.cross(P[4*i], P[4*i+1])
        l_group[i] = l_group[i]/np.max(l_group[i])
        m_group[i] = np.cross(P[4*i+2], P[4*i+3])
        m_group[i] = m_group[i]/np.max(m_group[i])
        B[i] = - l_group[i][2] * m_group[i][2]
        C[i] = np.array([l_group[i][0] * m_group[i][0], \
                         (l_group[i][0] * m_group[i][1] + l_group[i][1] * m_group[i][0])/2, \
                         l_group[i][1] * m_group[i][1], \
                         (l_group[i][0] * m_group[i][2] + l_group[i][2] * m_group[i][0])/2, \
                         (l_group[i][1] * m_group[i][2] + l_group[i][2] * m_group[i][1])/2])
    
    # print('l_group:', l_group)
    print('C:', C)
    print('B:', B)
    x = np.dot(np.linalg.inv(C), B)
    print('x:', x)
    x = x/np.max(x)
    print('x:', x)

    S = np.array([[x[0][0], x[1][0]/2],[x[1][0]/2, x[2][0]]])
    # print('S:', S) # 2 by 2
    U, D, V = np.linalg.svd(S)
    D =  np.diag(D)
    print('U:', U) # 2 by 2
    print('D:', D) # 2 by 2
    print('V:', V) # 2 by 2

    A = np.dot(np.dot(V,np.sqrt(D)),V.T)
    print('A:', A)
    v = np.dot(np.array([[x[3][0]/2,x[4][0]/2]]), np.linalg.inv(A.T))
    print('v:', v)

    H = np.eye(3)
    H[0][0:2] = A[0]
    H[1][0:2] = A[1]
    H[0][2] = v[0][0]
    H[1][2] = v[0][1]
    # print('H:', H)

    return H


# P = np.array([1,1,1]).T
# Q = np.array([1,2,1]).T
# R = np.array([1,3,1]).T
# S = np.array([2,1,1]).T
# Homography_vanish(P,Q,S,R)
# Homography_affine(P,Q,R,S)
# P0 = np.random.rand(20,2)
# Homography_1step(P0)

def load_point_for_one_step(args):
    if args.img_name == '1':
        # P0 = np.array([[x,y],[x,y]]) # 20 by 2
        P0 = np.array([[1011,1857],[957,2064],[1011,1857],[1269,1863],\
                        [1011,1857],[1269,1863],[1269,1863],[1218,2076],\
                        [1269,1863],[1218,2076],[1218,2076],[957,2064],\
                        [1218,2076],[957,2064],[957,2064],[1011,1857],\
                        [1512,1665],[1440,2085],[1512,1665],[1944,2592]])
    elif args.img_name == '2':
        pass
    elif args.img_name == '3':
        pass
    elif args.img_name == '4':
        pass
    else:
        raise Exception('Error: No such image!')
    return P0
   

def one_step0(img, args):
    print('img size:',img.shape) # img size: (1944, 2592, 3)
    P0 = load_point_for_one_step(args)
    # P0 = np.random.randint(20,size=(20,2))
    H = Homography_1step(P0)
    print('H:', H)
    # H_inv * X_i = X_w
    H_inv = np.linalg.inv(H)

    temp = np.zeros((img.shape[0],img.shape[1],3), dtype='uint8')
    image_target = img

    for i in range(0,(img.shape[0]-1)):
        for j in range(0,(img.shape[1]-1)):
            point_tmp = np.array([i, j, 1])
            trans_coord = np.array(np.dot(H,point_tmp))
            trans_coord = trans_coord/trans_coord[2]
            if (trans_coord[0] > 0) and (trans_coord[0] < image_target.shape[0]) and \
            (trans_coord[1] > 0) and (trans_coord[1] < image_target.shape[1]):
                temp[i][j] = image_target[math.floor(trans_coord[0]),math.floor(trans_coord[1])]   
    # return temp
    cv2.imwrite('../HW3Pics/onestep1.jpg',temp)

def one_step(img, args):
    print('img size:',img.shape) # img size: (1944, 2592, 3)
    P0 = load_point_for_one_step(args)
    # P0 = np.random.randint(20,size=(20,2))
    H = Homography_1step(P0)
    print('H:', H)
    # H_inv * X_i = X_w
    H_inv = np.linalg.inv(H)
    box_i = np.array([[1,1,1],[img.shape[1],1,1],[img.shape[0],1,1],[img.shape[1],img.shape[0],1]])
    # print(box_i.shape) # 4 by 3
    box_w = np.zeros((4,3))
    for i in range(box_i.shape[0]):
        box_w[i] = np.dot(H_inv,box_i[i])
        box_w[i] = box_w[i]//box_w[i][2]
    print('box_w:', box_w)
    xymin = np.min(box_w, axis=0)
    print('xymin:', xymin)
    xmin, ymin = xymin[0], xymin[1] 
    xymax = np.max(box_w, axis=0)
    xmax, ymax = xymax[0], xymax[1] 
    width = xmax - xmin
    height = ymax - ymin
    print('before scale', width, height) # 14513.0 4729.0 world plane
    scale_x = img.shape[1]/width
    scale_y = img.shape[0]/height
    print()
    width_ = math.floor(scale_x*width)
    height_ = math.floor(scale_y*height)
    print('after scale', width_, height_)

    img_out = np.zeros((height_, width_, 3))
    for h in range(height_):
        for w in range(width_):
            # scale back
            print('debug:', w/scale_x+xmin-1, h/scale_y+ymin-1)
            tmp = np.array([[w/scale_x+xmin-1],[h/scale_y+ymin-1],[1]])
            print(tmp)
            H_b = np.dot(H, tmp)
            print('H_b: ',H_b)
            x_i, y_i = int(H_b[0][0]//H_b[2][0]), int(H_b[1][0]//H_b[2][0])
            print(x_i, y_i)
            img_out[h][w] = img[x_i][y_i]

    # write output
    # img_out = np.zeros((height_, width_, 3))
    # img_out = np.zeros((height_, width_, 3))
    # for h in range(height_):
    #     for w in range(width_):
    #         # scale back
    #         tmp = np.array([[w/scale_x+xmin-1],[h/scale_y+ymin-1],[1]])
    #         print(tmp)
    #         H_b = np.dot(H, tmp)
    #         print('H_b: ',H_b)
    #         x_i, y_i = int(H_b[0][0]//H_b[2][0]), int(H_b[1][0]//H_b[2][0])
    #         print(x_i, y_i)
    #         img_out[h][w] = img[x_i][y_i]
    cv2.imwrite('../HW3Pics/onestep1.jpg',img_out)


def load_point_for_two_step(args):
    if args.img_name == '1':
        # world coordinate
        P_w, Q_w, S_w, R_w = np.array([1,1,1]), np.array([1,1,1]), np.array([1,1,1]), np.array([1,1,1])
        X_w = [P_w, Q_w, S_w, R_w]
        # point to point 
        P_i, Q_i, S_i, R_i = np.array([1,1,1]), np.array([1,1,1]), np.array([1,1,1]), np.array([1,1,1])
        X_i = [P_i, Q_i, S_i, R_i]
        # X_i = np.array([P_i, Q_i, S_i, R_i])
        # affine
        P_i2, Q_i2, S_i2, R_i2 = np.array([1,1,1]), np.array([1,1,1]), np.array([1,1,1]), np.array([1,1,1])
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
        pass
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
    # xymax = np.max(box_w, axis=0)
    # xmax, ymax = xymax[0], xymax[1] 
    # width = xmax - xmin
    # height = ymax - ymin
    # print('before scale', width, height)
    # scale_x = img.shape[1]/width
    # scale_y = img.shape[0]/height
    # width_ = math.floor(scale_x*width)
    # height_ = math.floor(scale_y*height)
    # print('after scale', width_, height_)

    # removing affine: x_h = H_p*x-xmin
    X_0 = []
    for pt in X_i2:
        tmp = np.dot(H_p,pt) # 3 by 1
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
    one_step(img1, args)
    # two_step(img1, args)



if __name__ == '__main__':
    main()
    # print()

