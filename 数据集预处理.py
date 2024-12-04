import os
import os.path
import time
import cv2
import numpy as np


def depth_rendered(Depth, savepath, use_depth_min=True):
    im_depth = cv2.imread(Depth, flags=cv2.IMREAD_UNCHANGED)
    depth_min, depth_max = 555, 810  # min < 621, max > 800
    different_min, different_max = 125, 1400  # min < 200, max > 1100
    use_min, use_max = [depth_min, depth_max] if use_depth_min else [different_min, different_max]
    im_depth[(im_depth < use_min)] = use_min
    im_depth[(im_depth > use_max)] = use_max
    im_depth = cv2.normalize(im_depth, None, 0, 255, cv2.NORM_MINMAX)
    im_depth = im_depth.astype(np.uint8)
    # im_depth = 255 - im_depth  # 灰度图颜色反转
    cv2.imwrite(savepath, im_depth)
    return  im_depth


def change_file_extension(path, old_ext, new_ext):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(old_ext):
                old_file = os.path.join(root, file)
                new_file = os.path.splitext(old_file)[0] + new_ext
                os.rename(old_file, new_file)


def RGB_D2RGBD(im_rgb, im_depth, savepath):
    im_rgb = np.array(im_rgb)  # 将图片转换为numpy数组
    im_depth = np.expand_dims(np.array(im_depth), axis=2)
    rgbd = np.concatenate((im_rgb, im_depth), axis=2)   # 融合
    cv2.imwrite(savepath, rgbd)  # 保存的RGBD图片文件名为RGB图片的文件名


def extract_number(file_name):
    return int(file_name.split("_")[0])  # 得到文件名中的数字部分


save_dataset_name, split_value = 'dataset/Snackbox_new', False
# save_dataset_name, split_value = 'dataset/Snackbox', True
dataset_name = r'your\dataset\path'
save_path = {'test': f"{save_dataset_name}/test/images",
             'train': f"{save_dataset_name}/train/images",
             'valid': f"{save_dataset_name}/valid/images"}  # 保存rgbd文件夹路径
depth_save_path = {'test': f"{save_dataset_name}/test/depth",
                   'train': f"{save_dataset_name}/train/depth",
                   'valid': f"{save_dataset_name}/valid/depth"}  # 保存depth文件夹路径
rgb_path = {'test': f"{dataset_name}/test/images",
            'train': f"{dataset_name}/train/images",
            'valid': f"{dataset_name}/valid/images"}  # rgb文件夹路径
depth_path = {'test': f"{dataset_name}/test/depth",
              'train': f"{dataset_name}/train/depth",
              'valid': f"{dataset_name}/valid/depth"}  # depth文件夹路径


if __name__ == '__main__':
    dataset_list = ['test', 'train', 'valid']
    for file in dataset_list:  # 遍历dataset_list文件夹内所有图片
        change_file_extension(rgb_path[file], ".jpg", ".png")
        change_file_extension(depth_path[file], ".jpg", ".png")
        if not os.path.exists(save_path[file]):
            os.makedirs(save_path[file])
        if not os.path.exists(depth_save_path[file]):
            os.makedirs(depth_save_path[file])
        start = time.time()
        print('start')
        for filename in os.listdir(depth_path[file]):
            rgb, depth = os.path.join(rgb_path[file], filename), os.path.join(depth_path[file], filename)
            depth_save = os.path.join(depth_save_path[file], filename)
            rgbd_save = os.path.join(save_path[file], filename)
            rgb_image = cv2.imread(rgb, cv2.IMREAD_UNCHANGED)
            index = extract_number(filename)
            depth_img = depth_rendered(depth, depth_save, False if index >= 5000 and split_value else True)  #
            RGB_D2RGBD(rgb_image, depth_img, rgbd_save)  # 1
        end = time.time()
        print(f"{file} cost ", end - start, "second")
