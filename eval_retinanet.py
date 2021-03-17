"""
05:
    mAP using the weighted average of precisions among classes: 0.7488
    mAP: 0.6741
08:
    mAP using the weighted average of precisions among classes: 0.6937
    mAP: 0.6220
"""
import os
import sys
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    __package__ = "retinanet"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from .preprocessing.csv_generator import CSVGenerator
from .utils.eval import evaluate
from retinanet import models


def create_generator(annotations, classes, image_min_side, image_max_side):
    """ Create generators for evaluation.
    """
    validation_generator = CSVGenerator(
        annotations,
        classes,
        image_min_side=image_min_side,
        image_max_side=image_max_side,
    )
    return validation_generator


if __name__ == '__main__':
    annotations = '/home/palm/PycharmProjects/seven2/anns/val_ann.csv'
    classes = '/home/palm/PycharmProjects/seven2/anns/classes.csv'
    image_min_side = 720
    image_max_side = 1280
    generator = create_generator(annotations, classes, image_min_side, image_max_side)

    model_path = '/home/palm/PycharmProjects/seven2/snapshots/infer_model_5.h5'
    model = models.load_model(model_path)

    average_precisions = evaluate(
        generator,
        model,
        iou_threshold=0.5,
        score_threshold=0.05,
        max_detections=100,
    )

    # print evaluation
    total_instances = []
    precisions = []
    for label, (average_precision, num_annotations) in average_precisions.items():
        print('{:.0f} instances of class'.format(num_annotations),
              generator.label_to_name(label), 'with average precision: {:.4f}'.format(average_precision))
        total_instances.append(num_annotations)
        precisions.append(average_precision)

    if sum(total_instances) == 0:
        print('No test instances found.')
        exit()

    print('mAP using the weighted average of precisions among classes: {:.4f}'.format(
        sum([a * b for a, b in zip(total_instances, precisions)]) / sum(total_instances)))
    print('mAP: {:.4f}'.format(sum(precisions) / sum(x > 0 for x in total_instances)))
