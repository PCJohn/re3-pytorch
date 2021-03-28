"""
Usage:

python3 make_labels.py /path/to/alov300++_frames/ /path/to/alov300++GT_txtFiles/

"""
import os
import re
import sys
import glob
import numpy as np


if __name__ == '__main__':
    frame_dir = sys.argv[1]
    label_dir = sys.argv[2]
    folders = glob.glob(os.path.join(frame_dir,'imagedata++','*','*'))
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
            cat,vid = folder.split('/')[-2:]
            label_files = [os.path.join(label_dir,'alov300++_rectangleAnnotation_full',cat,vid+'.ann')]
            image_files = sorted(glob.glob(os.path.join(folder,'*.jpg')))
            image_names.extend(image_files)
            for track_id, label_file in enumerate(label_files):
                raw_lines = [list(map(float, re.split("[,\s]", line.strip()))) for line in open(label_file)]
                lines = [[int(l[0]),int(l[3]),int(l[4]),(int(l[7])-int(l[3])),np.abs(int(l[8])-int(l[4]))] for l in raw_lines]
                #lines = [[int(l[0]),int(l[4]),int(l[3]),int(l[8])-int(l[4]),int(l[7])-int(l[3])] for l in raw_lines]

                ids = [l[0] for l in lines]
                #for frame_num in range(1,np.max(ids)):
                #    if not frame_num in ids:
                #        lines.append(list(sorted(lines,key=lambda l:np.abs(l[0]-frame_num))[0]))
                #        lines[-1][0] = frame_num
                
                lines.sort(key=lambda l:l[0]) # Sort lines by frame IDs
                lines = [l[1:] for l in lines] # Remove frame IDs from lines
                #lines = [list(map(float,l)) for l in lines] 
                #lines = [l for l in lines if (l[2] >= 0)]
                
                for k in range(len(lines)):
                    if lines[k][2] < 0:
                        lines[k][2] = -lines[k][2]
                        lines[k][0] -= lines[k][2]

                """
                import cv2
                id2file = {int(os.path.split(imf)[-1].split('.')[0]) : imf for imf in image_files}
                for k in range(len(ids)):
                    r = lines[k]
                    print(k,len(ids),ids,id2file.keys())
                    im = cv2.imread(id2file[ids[k]])
                    cv2.rectangle(im,(r[0],r[1]),(r[0]+r[2],r[1]+r[3]),(0,255,0),3)
                    cv2.imshow('im',im)
                    cv2.waitKey(0)
                    if lines[k][2] < 0:
                        import pdb; pdb.set_trace();
                """

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



