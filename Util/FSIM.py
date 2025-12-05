import cv2
import numpy as np
import os


def calculate_fsim(image1_path, image2_path):
    """
    Calculate the Feature Similarity Index (FSIM) between two images.

    Parameters:
    - image1_path: File path of the first image.
    - image2_path: File path of the second image.

    Returns:
    - fsim: FSIM value between the two images.
    """
    # Read images and convert to grayscale
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)

    if img1.shape != img2.shape:
        raise ValueError("The input two images must have the same dimensions")

    # Calculate phase consistency
    def phase_consistency(img1, img2):
        F1 = np.fft.fft2(img1)
        F2 = np.fft.fft2(img2)
        phase1 = np.angle(F1)
        phase2 = np.angle(F2)
        phi = (2 * (phase1 * phase2 + 1)) / (phase1 ** 2 + phase2 ** 2 + 1)
        return phi

    phi = phase_consistency(img1, img2)

    # Calculate gradient magnitude
    def gradient_magnitude(img):
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        return magnitude

    G1 = gradient_magnitude(img1)
    G2 = gradient_magnitude(img2)

    # Calculate gradient similarity
    eps = 1e-10
    sim_G = (2 * G1 * G2 + eps) / (G1 ** 2 + G2 ** 2 + eps)

    # Calculate FSIM
    fsim_map = phi * sim_G
    fsim = np.mean(fsim_map)

    return fsim


def calculate_fsim_for_folders(real_folder, fake_folder):
    """
    Calculate FSIM values for all image pairs between real and fake image folders.

    Assumptions:
    - Images in real_folder end with "png"
    - Corresponding fake images in fake_folder have filenames with "Real" replaced by "Fake"
    """
    fsim_values = []

    for real_filename in os.listdir(real_folder):
        if real_filename.endswith("png"):
            real_img_path = os.path.join(real_folder, real_filename)
            fake_filename = real_filename.replace("Real", "Fake")
            fake_img_path = os.path.join(fake_folder, fake_filename)

            # print(f"Processing: {real_img_path} -> {fake_img_path}")

            # Calculate FSIM for the image pair
            fsim_value = calculate_fsim(real_img_path, fake_img_path)
            fsim_values.append(fsim_value)

    return fsim_values