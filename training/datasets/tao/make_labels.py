import os
import sys
import glob
import json
import numpy as np


if __name__ == '__main__':
    TAO_train_imdir = sys.argv[1]
    TAO_train_labels = sys.argv[2]
    with open(TAO_train_labels,'r') as f:
        train_gt = json.load(f)    
    
    bboxes = []
    for vid in train_gt['videos']:
        vid_id = vid['id']
        vid_anns = [ann for ann in train_gt['annotations'] if (ann['video_id'] == vid['id'])]
        for ann in vid_anns:
            x,y,w,h = ann['bbox']
            xmin,ymin = x,y
            xmax,ymax = x+w,y+h
            bbox = [xmin,ymin,xmax,ymax]
            bboxes.append(bbox+
                            [vid_id,
                             ann['track_id'],
                             ann['image_id'],
                             ann['category_id'],
                             ann['iscrowd']])
    bboxes = np.array(bboxes)
    order = np.lexsort((bboxes[:, 6], bboxes[:, 5], bboxes[:, 4]))
    bboxes = bboxes[order, :]
    
    os.system('mkdir train')
    np.save('train/labels.npy',bboxes)
    
    image_paths = glob.glob(TAO_train_imdir+'/*/*/*/*')
    with open('train/image_paths.txt','w') as f:
        f.write('\n'.join(image_paths))
    

