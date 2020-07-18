import argparse
import logging
import sys
import time
import glob

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

    for file in glob.glob(args.inputdir+"/*.jpg"):
        name = file.split("/")[-1]
        print(name)
        image = common.read_imgfile(file, None, None)

        new_w = image.shape[1]-((image.shape[1]*70)//100)
        new_h = image.shape[0]-((image.shape[0]*70)//100)

        image = cv2.resize(image, (new_w,new_h), interpolation = cv2.INTER_AREA) 
        

        if image is None:
            logger.error('Image can not be read, path=%s' % image)
            sys.exit(-1)

        t = time.time()
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        elapsed = time.time() - t

        logger.info('inference image: %s in %.4f seconds.' % (image, elapsed))

        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        body_estimator = humans[0].body_parts[0]
        body_score_X   = humans[0].body_parts[0].x
        
        print(body_estimator)
        print(body_score_X)
        break

        # cv2.imshow('result', image)
        # cv2.imwrite(args.outputdir+"/"+name,image)

        # cv2.waitKey(0)