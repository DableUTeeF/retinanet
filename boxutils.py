import cv2
import numpy as np
import csv
from six import raise_from
import sys


color_list = np.array([0.466, 0.750, 0.325, 0.850, 0.098, 1.000, 0.929, 0.694, 0.125, 0.494, 0.184, 0.556,
                       1.000, 0.674, 0.188, 0.301, 0.745, 0.933, 0.635, 0.078, 0.184, 0.300, 0.300, 0.300,
                       0.600, 0.600, 0.600, 1.000, 0.000, 0.000, 1.000, 0.500, 0.000, 0.749, 0.749, 0.000,
                       0.000, 1.000, 0.000, 0.000, 0.000, 1.000, 0.667, 0.000, 1.000, 0.333, 0.667, 0.000,
                       0.333, 1.000, 0.000, 0.667, 0.333, 0.000, 0.667, 0.667, 0.000, 0.667, 1.000, 0.000,
                       1.000, 0.333, 0.000, 1.000, 0.667, 0.000, 1.000, 1.000, 0.000, 0.000, 0.333, 0.500,
                       0.000, 0.667, 0.500, 0.000, 1.000, 0.500, 0.333, 0.000, 0.500, 0.333, 0.333, 0.500,
                       0.333, 0.667, 0.500, 0.333, 1.000, 0.500, 0.667, 0.000, 0.500, 0.667, 0.333, 0.500,
                       0.667, 0.667, 0.500, 0.667, 1.000, 0.500, 1.000, 0.000, 0.500, 1.000, 0.333, 0.500,
                       1.000, 0.667, 0.500, 1.000, 1.000, 0.500, 0.000, 0.333, 1.000, 0.000, 0.667, 1.000,
                       0.000, 1.000, 1.000, 0.333, 0.000, 1.000, 0.333, 0.333, 1.000, 0.333, 0.667, 1.000,
                       0.333, 1.000, 1.000, 0.667, 0.000, 1.000, 0.667, 0.333, 1.000, 0.667, 0.667, 1.000,
                       0.667, 1.000, 1.000, 1.000, 0.000, 1.000, 1.000, 0.333, 1.000, 1.000, 0.667, 1.000,
                       0.167, 0.000, 0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000,
                       0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000, 0.333, 0.000,
                       0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000, 0.000, 1.000, 0.000,
                       0.000, 0.000, 0.167, 0.000, 0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667,
                       0.000, 0.000, 0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.143, 0.143, 0.143,
                       0.286, 0.286, 0.286, 0.429, 0.429, 0.429, 0.571, 0.571, 0.571, 0.714, 0.714, 0.714,
                       0.857, 0.857, 0.857, 0.000, 0.447, 0.741, 0.50, 0.5, 0]).astype(np.float32)
color_list = color_list.reshape((-1, 3)) * 255
color_list = np.concatenate((color_list, color_list/2))

colors = [(color_list[_]).astype(np.uint8) for _ in range(len(color_list))]
colors = np.array(colors, dtype=np.uint8).reshape(len(colors), 1, 1, 3)


def add_bbox(img, bbox, cat, labels, conf=1, show_txt=True, color=None):
    # bbox = np.array(bbox, dtype=np.int32)
    # cat = (int(cat) + 1) % 80
    cat = int(cat)
    c = colors[cat % 158][0][0].tolist() if color is None else color
    txt = '{} - {:.1f}'.format(labels[cat], conf)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), c, 2)
    if show_txt:
        cv2.rectangle(img,
                      (bbox[0], bbox[1] - cat_size[1] - 2),
                      (bbox[0] + cat_size[0], bbox[1] - 2), c, -1)
        cv2.putText(img, txt, (bbox[0], bbox[1] - 2),
                    font, 0.5, (0, 0, 0) if cat < 80 else (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
    return img




def _parse(value, function, fmt):
    """
    Parse a string into a value, and format a nice ValueError if it fails.

    Returns `function(value)`.
    Any `ValueError` raised is catched and a new `ValueError` is raised
    with message `fmt.format(e)`, where `e` is the caught `ValueError`.
    """
    try:
        return function(value)
    except ValueError as e:
        raise_from(ValueError(fmt.format(e)), None)


def _read_classes(csv_reader):
    """ Parse the classes file given by csv_reader.
    """
    result = {}
    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise_from(ValueError('line {}: format should be \'class_name,class_id\''.format(line)), None)
        class_id = _parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result


def _read_annotations(csv_reader, classes):
    """ Read annotations from the csv_reader.
    """
    result = {}
    for line, row in enumerate(csv_reader):
        line += 1

        try:
            img_file, x1, y1, x2, y2, class_name = row[:6]
        except ValueError:
            raise_from(ValueError(
                'line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(line)),
                       None)

        if img_file not in result:
            result[img_file] = []

        # If a row contains only an image path, it's an image without annotations.
        if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
            continue

        x1 = _parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
        y1 = _parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
        x2 = _parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
        y2 = _parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))

        # Check that the bounding box is valid.
        if x2 <= x1:
            raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
        if y2 <= y1:
            raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

        # check if the current class name is correctly present
        if class_name not in classes:
            raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))

        result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name})
    return result


def _open_for_csv(path):
    """ Open a file with flags suitable for csv.reader.

    This is different for python2 it means with mode 'rb',
    for python3 this means 'r' with "universal newlines".
    """
    if sys.version_info[0] < 3:
        return open(path, 'rb')
    else:
        return open(path, 'r', newline='')


def create_csv_training_instances(train_csv, class_csv, with_wh=False):
    with _open_for_csv(class_csv) as file:
        classes = _read_classes(csv.reader(file, delimiter=','))
    with _open_for_csv(train_csv) as file:
        train_image_data = _read_annotations(csv.reader(file, delimiter=','), classes)
    train_ints = []
    labels = list(classes)
    max_box_per_image = 0
    for k in train_image_data:
        image_data = train_image_data[k]
        ints = {'filename': k, 'object': []}
        for i, obj in enumerate(image_data):
            o = {'xmin': obj['x1'], 'xmax': obj['x2'], 'ymin': obj['y1'], 'ymax': obj['y2'], 'name': obj['class']}
            if with_wh:
                x = cv2.imread(k)
                height, width, _ = x.shape
                o['width'] = width
                o['height'] = height
            ints['object'].append(o)
            if i + 1 > max_box_per_image:
                max_box_per_image = i + 1
        train_ints.append(ints)

    return train_ints, sorted(labels), max_box_per_image

