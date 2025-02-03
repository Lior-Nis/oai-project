import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.ndimage as sc
from skimage.draw import line_nd
from tqdm import tqdm
from PIL import Image

def list_files_in_folder(folder, with_path=True):
    # if folder contains folders of files
    folders = [os.path.join(folder, f) for f in os.listdir(folder)]
    if not os.path.isfile(folders[0]):
        files = []
        for folder in folders:
            if os.path.isdir(folder):
                if with_path:
                    files += [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.tif')]
                else:
                    files += [f for f in os.listdir(folder) if f.endswith('.tif')]
        return files

    # else if folder contains files
    if with_path:
        return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.tif')]

    return [f for f in os.listdir(folder) if f.endswith('.tif')]

def parse_minute_file(mnt_file_path):
    mnt = np.loadtxt(mnt_file_path)[:, :4]
    mnt[:, -1] = mnt[:, -1] * np.pi / 180
    return mnt

def parse_minute_file_old(mnt_file_path):
    with open(mnt_file_path, 'r') as mnt_file:
        mnt_list = [line.strip().split(' ') for line in mnt_file.readlines()]
        mnt_out = []
        for mnt in mnt_list:
            p1 = (int(mnt[1]), int(mnt[2]))
            p2 = (int(mnt[1]) + int(mnt[4]), int(mnt[2]) + int(mnt[4]))
            type = int(mnt[0])
            mnt_dict = {"type": type,
                        "p1": p1,
                        "p2": p2}
            mnt_out.append(mnt_dict)
        return mnt_out

def create_minutiae_map(mnts, size=(768, 832)):
    minutiae_map = np.zeros(size, dtype=np.uint8)
    x = mnts[:, 1].astype(np.int32).tolist()
    y = mnts[:, 2].astype(np.int32).tolist()
    minutiae_map[y, x] = 255
    return minutiae_map

def create_orientation_map(mnts, size=(768, 832), ori_length=15):
    orientation_map = np.zeros(size, dtype=np.uint8)
    for x, y, ori in zip(mnts[:, 1], mnts[:, 2], mnts[:, 3]):
        x, y = int(x), int(y)
        x1, y1 = x, y
        x2, y2 = x + ori_length * np.cos(ori), y - ori_length * np.sin(ori)
        line_idx = line_nd((y1, x1), (y2, x2), endpoint=True)
        orientation_map[line_idx] = 255
    return orientation_map

def create_map_scipy(mnts, size=(768, 832), num_of_maps=3, ori_length=15, mnt_sigma=9, ori_sigma=3,
                     mnt_gain=60, ori_gain=3, include_singular=False):
    maps = []
    if include_singular:  # include core and delta points
        types = [[1], [2], [4, 5]]
    else:
        types = [[1], [2], [-1]]

    for idx in range(num_of_maps):
        minutiae_map = create_minutiae_map(mnts[mnts[:, 0] == types[idx][0]], size)
        orientation_map = create_orientation_map(mnts[mnts[:, 0] == types[idx][0]], size, ori_length)
        map_blur = (sc.gaussian_filter(minutiae_map, sigma=np.sqrt(mnt_sigma))[:, :, np.newaxis] * mnt_gain).astype(int)
        map_ori_blur = (sc.gaussian_filter(orientation_map, sigma=np.sqrt(ori_sigma))[:, :, np.newaxis] * ori_gain).astype(int)
        maps.append(map_blur + map_ori_blur)

    output = np.concatenate(maps, axis=-1)
    output[output > 255] = 255
    output = output.astype(np.uint8)
    return output


def create_map_scipy_without_ori(mnt_dict_list, size=(768, 832), num_of_maps=3):
    maps = []
    types = [[1], [2], [4, 5]]
    for idx in range(num_of_maps):
        map = np.zeros(size)
        x = [mnt_dict['p1'][0] for mnt_dict in mnt_dict_list if mnt_dict['type'] in types[idx]]
        y = [mnt_dict['p1'][1] for mnt_dict in mnt_dict_list if mnt_dict['type'] in types[idx]]
        map[[y, x]] = 1
        maps.append(sc.gaussian_filter(map, sigma=np.sqrt(3))[:, :, np.newaxis] * 20)
    output = np.concatenate(maps, axis=-1)
    return output

def create_map_dir(min_folder, img_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in tqdm(os.listdir(min_folder)):
      try:
        img = Image.open(os.path.join(img_folder, filename.split(".")[0] + ".tif"))
        width, height = img.size
        txt_path = os.path.join(min_folder, filename)
        mnt = parse_minute_file(txt_path)
        map = create_map_scipy(mnt, include_singular=False, size=(height, width))
        map_filename = os.path.join(output_folder, os.path.splitext(filename)[0] + ".png")
        plt.imsave(map_filename, map)
      except Exception as e:
        # print(f"Exception occurred while creating minutiae maps: {e}")
        print("File name: ", filename)
        # raise e


def main():
    # input arguments
    imgs_path = '/content/oai_project/data_samples/scans'
    txts_path = '/content/oai_project/data_samples/minutiae'
    output_path = '/content/min_maps_fixed'

    # create output folder if not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    create_map_dir(txts_path, imgs_path, output_path)


if __name__ == "__main__":
    main()
