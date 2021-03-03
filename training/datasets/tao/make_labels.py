import os
import json
import numpy as np


if __name__ == '__main__':
    TAO_train_imdir = '/home/prithvi/data/tracking/1-TAO_TRAIN'
    TAO_train_labels = '/home/prithvi/data/tracking/TAOLabels/TAO_train/gt/gt.json'
    train_gt = json.load(open(TAO_train_labels,'r'))    
    
    bboxes = []
    for vid in train_gt['videos']:
        vid_id = vid['id']
        vid_anns = [ann for ann in train_gt['annotations'] if (ann['video_id'] == vid['id'])]
        for ann in vid_anns:
            bboxes.append(ann['bbox']+
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
    
    #os.system('mkdir val')
    #np.save('val/labels.npy',bboxes)




