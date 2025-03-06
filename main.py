import cv2
import numpy as np

def laplacian_pyramid(img, levels=5):
    gaussian_pyr = [img]
    for _ in range(levels):
        img = cv2.pyrDown(img)
        gaussian_pyr.append(img)
    
    laplacian_pyr = [gaussian_pyr[-1]]
    for i in range(levels, 0, -1):
        size = (gaussian_pyr[i - 1].shape[1], gaussian_pyr[i - 1].shape[0])
        laplacian = cv2.subtract(gaussian_pyr[i - 1], cv2.pyrUp(gaussian_pyr[i], dstsize=size))
        laplacian_pyr.append(laplacian)
    
    return laplacian_pyr

def reconstruct_from_laplacian_pyramid(laplacian_pyr):
    img = laplacian_pyr[0]
    for i in range(1, len(laplacian_pyr)):
        size = (laplacian_pyr[i].shape[1], laplacian_pyr[i].shape[0])
        img = cv2.pyrUp(img, dstsize=size) + laplacian_pyr[i]
    return img

def exposure_fusion(images):
    levels = 5
    laplacian_pyramids = [laplacian_pyramid(img, levels) for img in images]
    
    fused_pyr = []
    for level in range(levels + 1):
        fused_pyr.append(np.max([lp[level] for lp in laplacian_pyramids], axis=0))
    
    fused_img = reconstruct_from_laplacian_pyramid(fused_pyr)
    fused_img = np.clip(fused_img, 0, 255).astype(np.uint8)
    return fused_img

def main():
    img1 = cv2.imread('image1.jpg')
    img2 = cv2.imread('image2.jpg')
    img3 = cv2.imread('image3.jpg')
    
    images = [img1, img2, img3]
    fused_image = exposure_fusion(images)
    
    cv2.imwrite('fused_image.jpg', fused_image)
    cv2.imshow('Fused Image', fused_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
