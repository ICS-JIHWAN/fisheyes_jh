# train --- train_source_image  : # 2194(Flatten)
#        |- train_source_gt     : # 2194(Mask)
#        |- train_source.csv    : # id & image path & gt path
#        |- train_target_image  : # 2923(Fish Eyes)
#        |- train_target.csv    : # id & image path
# |===========================|
# val------ val_source_image    : # 466(Flatten)
#        |- val_source_gt       : # 466(Mask)
#        |- val_source.csv      : id & image path & gt path
# |===========================|
# test ---- test_image          : # 1898(Fish Eyes)
#        |- test.csv            : id & image path

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

class2color = {
    0: (128, 128, 128), # Road
    1: (192, 192, 192), # Sidewalk
    2: (127, 0, 255),   # Construction
    3: (255, 0, 255),   # Fence
    4: (255, 153, 255), # Pole
    5: (102, 102, 255), # Traffic light
    6: (51, 153, 255),  # Traffic sign
    7: (0, 255, 128),   # Nature
    8: (255, 255, 102), # Sky
    9: (102, 0, 0),     # Person
    10: (255, 51, 51),  # Rider
    11: (0, 179, 255)   # Car
}

def save_csvFile(root, df, filename: str):
    df_dict = dict()
    for k in df.keys():
        if k == 'id':
            df_dict[k] = sorted(list(map(lambda x: x, df[k])))
            continue
        df_dict[k] = sorted(list(map(lambda x: x[2:], df[k])))

    df_A = pd.DataFrame(df_dict)

    try:
        if not os.path.isfile(os.path.join(root, filename)):
            df_A.to_csv(os.path.join(root, filename), index=False)
    except:
        ValueError('No input filename!!')


root = '/storage/jhchoi/fish_eyes'

# Read .csv file
read_file = 'cvt_'  # ['full_path_', 'cvt_', '']
train_source = pd.read_csv(os.path.join(root, 'csvfile/{}train_source.csv'.format(read_file)))
train_target = pd.read_csv(os.path.join(root, 'csvfile/{}train_target.csv'.format(read_file)))
val_source = pd.read_csv(os.path.join(root, 'csvfile/{}val_source.csv'.format(read_file)))
test = pd.read_csv(os.path.join(root, 'csvfile/{}test.csv'.format(read_file)))

# To modify the file path in .csv file
save = False
if save:
    cvt_file_name = 'cvt_'
    save_csvFile(os.path.join(root, 'csvfile'), train_source, '{}train_source.csv'.format(cvt_file_name))
    save_csvFile(os.path.join(root, 'csvfile'), train_target, '{}train_target.csv'.format(cvt_file_name))
    save_csvFile(os.path.join(root, 'csvfile'), val_source, '{}val_source.csv'.format(cvt_file_name))
    save_csvFile(os.path.join(root, 'csvfile'), test, '{}test.csv'.format(cvt_file_name))

size_dict = dict()
for idx, paths in enumerate(train_source[['img_path', 'gt_path']].values.tolist()):
    img = cv2.imread(os.path.join(root, paths[0]))
    gt = cv2.imread(os.path.join(root, paths[1]))

    h, w, _ = img.shape
    if '{}x{}'.format(h, w) in size_dict.keys():
        size_dict['{}x{}'.format(h, w)] += 1
    else:
        size_dict['{}x{}'.format(h, w)] = 1


    # blending
    blending_save = True
    if blending_save:
        for cls in class2color.keys():
            pixel_index = np.where(gt == cls)
            gt[pixel_index[0], pixel_index[1]] = class2color[cls]

        alpha = 0.5
        b_image = cv2.addWeighted(img, 1.0 - alpha, gt, alpha, 0)
        cv2.imshow('12', b_image)
        cv2.waitKey()
        cv2.destroyWindow('12')
        cv2.imwrite(os.path.join(root, '{}_blendingImg.png'.format(train_source['id'][idx])), b_image)

for idx, paths in enumerate(test[['img_path']].values.tolist()):
    img = cv2.imread(os.path.join(root, paths[0]))

    h, w, _ = img.shape
    if '{}x{}'.format(h, w) in size_dict.keys():
        size_dict['{}x{}'.format(h, w)] += 1
    else:
        size_dict['{}x{}'.format(h, w)] = 1

for idx, paths in enumerate(val_source[['img_path']].values.tolist()):
    img = cv2.imread(os.path.join(root, paths[0]))

    h, w, _ = img.shape
    if '{}x{}'.format(h, w) in size_dict.keys():
        size_dict['{}x{}'.format(h, w)] += 1
    else:
        size_dict['{}x{}'.format(h, w)] = 1

for idx, paths in enumerate(train_target[['img_path']].values.tolist()):
    img = cv2.imread(os.path.join(root, paths[0]))

    h, w, _ = img.shape
    if '{}x{}'.format(h, w) in size_dict.keys():
        size_dict['{}x{}'.format(h, w)] += 1
    else:
        size_dict['{}x{}'.format(h, w)] = 1

print(size_dict)

