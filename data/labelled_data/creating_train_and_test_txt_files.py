import os

"""
Create train.txt and test.txt for training

Algorithm:
    Setting up full paths --> List of paths -->
    --> Extracting 15% of paths to save into test.txt file -->
    --> Writing paths into train and test txt files
    
Return:
    train.txt, test.txt
"""

full_path_to_images = r'C:\Users\15734\PycharmProjects\nike_series\data\images\Nike SB Dunk Low StrangeLove'
test_size = 0.15
os.chdir(full_path_to_images)


# get image paths
img_paths = []
for current_dir, dirs, files in os.walk('.'):
    for f in files:
        # Checking if filename ends with '.jpg' or '.png'
        if f.endswith('.jpg') or f.endswith('.png'):
            print(f)
            path_to_save_into_txt_files = full_path_to_images + '/' + f
            img_paths.append(path_to_save_into_txt_files + '\n')

# train test split
img_train = img_paths[int(len(img_paths) * test_size):]
img_test = img_paths[:int(len(img_paths) * test_size)]

# write image paths to train.txt and test.txt
with open('train.txt', 'w') as f:
    for path in img_train:
        f.write(path)
f.close()

with open('test.txt', 'w') as f:
    for path in img_test:
        f.write(path)
f.close()
