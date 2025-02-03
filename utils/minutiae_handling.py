import subprocess
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.ndimage as sc
from skimage.draw import line_nd
from tqdm import tqdm
import glob
from PIL import Image

# Define the path to the mindtct and bozorth3 executables
bozorth_executable = "./bozorth3"


def get_file_name_and_ext(image_file):
    full_file_name = os.path.basename(image_file)
    file_name, file_ext = os.path.splitext(full_file_name)
    return full_file_name, file_name, file_ext[1:]


def call_mindtct(file_in, file_out):
    mindtct_path = "./mindtct"
    argv1 = f'"{file_in}"'
    argv2 = f'"{file_out}"'
    arguments = f'-m1 {argv1} {argv2}'

    try:
        exe_process = subprocess.Popen(f'{mindtct_path} {arguments}', shell=True, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
        exe_process.wait()

        stdout, stderr = exe_process.communicate()
        if stderr:
            print(f"Error running mindtct: {stderr.decode('utf-8')}")

    except Exception as e:
        print(f"Exception occurred while running mindtct: {e}")


def convert_image_to_png(image_file, output_dir):
    image = Image.open(image_file)

    # Convert the image to 8-bit grayscale
    if image.mode != 'L':
        image = image.convert("L")

    # Save the converted image as PNG
    file_name, _ = os.path.splitext(image_file)
    output_file = str(file_name).split("/")[-1] + "_8bit.png"
    output_path = os.path.join(output_dir, output_file)
    image.save(output_path, 'PNG')

    image.close()
    return output_path


def extract_dir(target_dir, output_dir=None, xyt_output=False, keep_all=False):
    if output_dir is None:
        output_dir = os.path.normpath(target_dir + os.sep + os.pardir) + "\\" + os.path.basename(
            os.path.normpath(target_dir)) + "_min\\"

    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
    try:
        for root, dirs, files in os.walk(target_dir):
            for file in tqdm(files):
                if file.lower().endswith(
                        ('.bmp', '.jpg', '.jpeg', '.png', '.tif')):  # Add more image extensions as needed
                    image_path = os.path.join(root, file)
                    # output_subdir = os.path.join(output_dir, os.path.splitext(file)[0])
                    output_subdir = output_dir + "/"

                    # Run mindtct and output minutiae to the subdirectory
                    call_mindtct(image_path, output_subdir + os.path.splitext(file)[0])

                    if not keep_all:
                        output_file_prefix = os.path.join(output_subdir, os.path.splitext(file)[0])
                        expected_files = ['dm', 'hcm', 'lcm', 'lfm', 'qm', 'brw']
                        if xyt_output:
                            expected_files.append('min')
                        else:
                            expected_files.append('xyt')
                        for ext in expected_files:
                            file_to_remove = output_file_prefix + '.' + ext
                            if os.path.exists(file_to_remove):
                                os.remove(file_to_remove)
    except Exception as e:
        # print(f"Exception occurred while creating minutiae maps: {e}")
        print("File name: ", file)
        raise e


def convert_minutiae_to_txt(min_file_path, output_dir):
    with open(min_file_path, 'r') as min_file:
        lines = min_file.readlines()

        # Extract the name of the .min file
        file_name = os.path.splitext(os.path.basename(min_file_path))[0]
        output_file_path = os.path.join(output_dir, file_name + ".txt")

        # Create a new .txt file for minutiae data
        with open(output_file_path, 'w') as output_file:
            for line in lines[3:]:  # Skip first three lines (header)
                line = line.strip()
                if line:
                    # Split the line into fields based on colons
                    fields = line.split(':')
                    # quality = float(fields[3].strip())
                    # Check if quality is above 0.2 before writing to the file
                    # if quality >= 0.2:
                    if True:
                        # Extract relevant information
                        mn_type = fields[4].strip()
                        x, y = map(int, fields[1].strip().split(','))
                        direction_unit = int(fields[2].strip())
                        # Convert direction_unit to degrees (0-360)
                        orientation = (90 - (11.25 * direction_unit)) % 360

                        minutiae_type = 1 if 'BIF' in mn_type else 2

                        # Write the minutiae data in the desired format to the new file
                        output_file.write(f"{minutiae_type} {x} {y} {orientation}\n")


def convert_all_minutiae_files(source_dir, output_dir=None):
    if output_dir is None:
        output_dir = os.path.normpath(source_dir + os.sep + os.pardir) + "\\" + os.path.basename(os.path.normpath(source_dir)) + "txt\\"

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Iterate through all subdirectories in the source directory
        for root, _, files in os.walk(source_dir):
            for f in tqdm(files):
                min_file_path = os.path.join(root, f)
                convert_minutiae_to_txt(min_file_path, output_dir)
    except Exception as e:
        # print(f"Exception occurred while creating minutiae maps: {e}")
        print("File name: ", f)
        raise e


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
            # Get the base name without extension.
            base = os.path.splitext(filename)[0]
            # Find any file in img_folder that starts with the same base name.
            img_candidates = glob.glob(os.path.join(img_folder, base + ".*"))
            if not img_candidates:
                raise FileNotFoundError(f"No image file found for {base} in {img_folder}")
            # Use the first matching file.
            img_path = img_candidates[0]
            img = Image.open(img_path)
            width, height = img.size

            txt_path = os.path.join(min_folder, filename)
            mnt = parse_minute_file(txt_path)
            map_img = create_map_scipy(mnt, include_singular=False, size=(height, width))

            # Save the map with a .png extension.
            map_filename = os.path.join(output_folder, base + ".png")
            plt.imsave(map_filename, map_img)
        except Exception as e:
            print(f"Exception occurred while creating minutiae maps: {e}")
            print("File name:", filename)


def run_bozorth3(probe_file, gallery_file):
    command = [bozorth_executable, probe_file, gallery_file]

    try:
        result = subprocess.run(command, capture_output=True, text=True)
        # Check if Bozorth3 successfully computes the score
        if result.returncode == 0:
            # Parse the output to extract the matching score
            score_line = result.stdout.strip().split('\n')[-1]  # Extract the last line
            score = float(score_line.split()[0])  # Assuming score is in the first column
            return score
        else:
            return None  # Return None if Bozorth3 failed for this pair
    except subprocess.CalledProcessError as e:
        print(f"Error running Bozorth3: {e}")
        return None

    except Exception as e:
        print(f"Exception occurred while running Bozorth3: {e}")
        return None


def bozorth3_on_pairs(dir1, dir2):
    matching_scores = []
    for file1 in os.listdir(dir1):
        # Run Bozorth3 on the pair of minutiae files
        score = run_bozorth3(os.path.join(dir1, file1), os.path.join(dir2, file1))
        matching_scores.append(score)
    return matching_scores


def calculate_and_plot_scores(matching_scores):
    scores = np.array(matching_scores)

    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=50, color='blue', edgecolor='black')
    plt.axvline(x=40, color='red', linestyle='--', label='Threshold')
    plt.xlabel('Matching Scores')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Matching Scores')
    # plt.savefig(os.path.join(fake_dir, 'score_distribution.png'))
    # plt.close()
    plt.show()

    print(f"\nStatistics for Experiment:")
    print(f"Mean score: {np.mean(scores):.2f}")
    print(f"Max score: {np.max(scores):.2f}")
    print(f"Percent of scores above 40: {(len(scores[scores > 40]) / len(scores) * 100):.2f}%")