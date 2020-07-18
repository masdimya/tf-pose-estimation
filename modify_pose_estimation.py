import argparse
import logging
import sys
import time
import glob
import json

from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimatorRun')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)



def get_human_point(humans):
    pose_json = {}
    pose_json['parts'] = [{'id': i,'x':0,'y':0} for i in range(19)]
    
    for i in humans[0].body_parts:
        pose_json['parts'][i]['x'] = humans[0].body_parts[i].x
        pose_json['parts'][i]['y'] = humans[0].body_parts[i].y
    
    return pose_json



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--inputdir',  type=str)
    parser.add_argument('--outputdir', type=str)
    parser.add_argument('--model', type=str, default='cmu',
                        help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. '
                             'default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    args = parser.parse_args()

    w, h = model_wh(args.resize)
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    # estimate human poses from a single image !

    human_point_json = {"frame" : []}

    for file in sorted(glob.glob(args.inputdir+"/*.jpg")):
        name = file.split("/")[-1]
        print(name)
    #     image = common.read_imgfile(file, None, None)

    #     new_w = image.shape[1]-((image.shape[1]*70)//100)
    #     new_h = image.shape[0]-((image.shape[0]*70)//100)

    #     image = cv2.resize(image, (new_w,new_h), interpolation = cv2.INTER_AREA) 
        

    #     if image is None:
    #         logger.error('Image can not be read, path=%s' % image)
    #         sys.exit(-1)

    #     t = time.time()
    #     humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
    #     elapsed = time.time() - t

    #     logger.info('inference image: %s in %.4f seconds.' % (image, elapsed))

    #     image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        
    #     human_point_json['frame'].append(get_human_point(humans))
        
    #     break

    # f = open(args.outputdir+"/"+name[:-4]+".json",'w+')
    # str_obj = json.dumps(human_point_json)
    # f.write(str_obj+'\n')
    # f.close()