import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from retinanet.utils.colors import label_color
from retinanet import models
import cv2
import numpy as np
import time
from boxutils import add_bbox


if __name__ == '__main__':
    labels_to_names = [x.split(',')[0] for x in open('/home/palm/PycharmProjects/seven2/anns/c.csv').read().split('\n')[:-1]]
    model_path = '/home/palm/PycharmProjects/seven2/snapshots/infer_model_5.h5'
    model = models.load_model(model_path)

    dst = '/home/palm/PycharmProjects/seven/predict/1'
    path = '/home/palm/PycharmProjects/seven/data1/1'
    pad = 0
    for image_name in os.listdir(path):
        p = os.path.join(path, image_name)

        image = read_image_bgr(p)

        # copy to draw on
        draw = image.copy()

        # preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image, min_side=720, max_side=1280)

        # process image
        start = time.time()
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        print("processing time: ", time.time() - start)

        # correct for image scale
        boxes /= scale

        # visualize detections
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
            if score < 0.5:
                break

            b = box.astype(int)
            # draw = add_bbox(draw, b, label, labels_to_names, score)

        # os.makedirs(dst, exist_ok=True)
        # cv2.imwrite(os.path.join(dst, image_name), draw)
