import torch
import cv2

def rgb_to_hsv_tensor(img_tensor, device):

    def rgb_to_hsv(image):
        image_cpu = image.cpu() if image.is_cuda else image
        image_np = image_cpu.numpy().transpose(1, 2, 0)
        image_np_hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        return torch.tensor(image_np_hsv.transpose(2, 0, 1), device=device, dtype=torch.float32)
    
    return torch.stack([rgb_to_hsv(image) for image in img_tensor])


def rgb_to_ycbcr_tensor(img_tensor, device):

    def rgb_to_ycbcr(image):
        image_cpu = image.cpu() if image.is_cuda else image
        image_np = image_cpu.numpy().transpose(1, 2, 0)
        image_np_ycbcr = cv2.cvtColor(image_np, cv2.COLOR_RGB2YCrCb)
        return torch.tensor(image_np_ycbcr.transpose(2, 0, 1), device=device, dtype=torch.float32)
    
    return torch.stack([rgb_to_ycbcr(image) for image in img_tensor])
