import numpy as np
import cv2

def WarpPerspectiveMatrix(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4
    
    nums = src.shape[0]
    A = np.zeros((2 * nums, 8))
    B = np.zeros((2 * nums, 1))
    
    for i in range(nums):
        A_i = src[i, :]
        B_i = dst[i, :]
        
        A[2 * i] = [A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
        B[2 * i] = B_i[0]
        
        A[2 * i + 1] = [0, 0, 0, A_i[0], A_i[1], 1, -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        B[2 * i + 1] = B_i[1]
    
    # Convert A and B to np.array for matrix operations
    A = np.array(A)
    B = np.array(B)
    
    # Calculate the warp matrix by solving the system A * warpMatrix = B
    warpMatrix = np.linalg.inv(A.T @ A) @ A.T @ B  
    
    # Post-process the result to form a 3x3 transformation matrix
    warpMatrix = np.append(warpMatrix, 1).reshape((3, 3))
    
    return warpMatrix
if __name__ == '__main__':
    print('warpMatrix')
    src = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
    src = np.array(src)
    
    dst = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]
    dst = np.array(dst)
    
    warpMatrix = WarpPerspectiveMatrix(src, dst)
    print(warpMatrix)
    # use open cv:
    # m = cv2.getPerspectiveTransform(src, dst)
    # result = cv2.warpPerspective(source, m, (x, y))