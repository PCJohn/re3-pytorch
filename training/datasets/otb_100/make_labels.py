import os
import re
import sys
import glob
import numpy as np


if __name__ == '__main__':
    data_dir = sys.argv[1]
    folders = glob.glob(os.path.join(data_dir,'*'))
    np.random.shuffle(folders)
    train_split = int(0.8*len(folders))
    train_folders, val_folders = folders[:train_split], folders[train_split:]
    
    for split,folder_list in zip(['train','val'],[train_folders,val_folders]):
        im_id = 0
        video_id = 0
        image_names = []
        all_lines = []
        for folder in sorted(folder_list):
            im_id_start = im_id
            label_files = sorted(glob.glob(folder + "/groundtruth_rect*.txt"))
            image_files = sorted(glob.glob(folder + "/img/*.jpg"))
            image_names.extend(image_files)
            for track_id, label_file in enumerate(label_files):
                lines = [list(map(float, re.split("[,\s]", line.strip()))) for line in open(label_file)]
                
                if len(lines) != len(image_files):
                    print("not equal", len(lines), len(image_files), folder)
                for line in lines:
                    line.extend([video_id, track_id, im_id])
                    im_id += 1
                all_lines.extend(lines)
                im_id = im_id_start
                video_id += 1
            im_id += len(image_files)

        all_lines = np.array(all_lines)
        all_lines[:, 2] += all_lines[:, 0]
        all_lines[:, 3] += all_lines[:, 1]
        os.system('mkdir labels')
        os.system('mkdir '+os.path.join('labels',split))
        np.save(os.path.join('labels',split,'labels.npy'), all_lines)
        ff = open(os.path.join('labels',split,'image_names.txt'), 'w')
        ff.write('\n'.join(image_names))
        ff.close()
        print('done')
        print('num labels', all_lines.shape[0])
        print('num images', len(image_names))



