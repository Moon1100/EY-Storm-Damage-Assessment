import os
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import cv2
import numpy as np


def parse_yolo_annotation(annotations_file, image_height, image_width):
    """
    Parse YOLO annotation file and return list of BoundingBoxesOnImage.
    """
    bounding_boxes = []

    with open(annotations_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            class_id, x_center, y_center, width, height = map(float, line.split())
            x1 = (x_center - width / 2) * image_width
            y1 = (y_center - height / 2) * image_height
            x2 = (x_center + width / 2) * image_width
            y2 = (y_center + height / 2) * image_height
            bounding_boxes.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=class_id))

    return BoundingBoxesOnImage(bounding_boxes, shape=(image_height, image_width, 3))


def revert_coordinate(x1, y1, x2, y2, image_height, image_width):

    width = (x2 - x1)/image_width
    x_center = (x1/image_width) + width / 2

    height = (y2 - y1)/image_height
    y_center = (y1/image_height) + height / 2

    return x_center, y_center, width, height


### INPUT: dataset path, automaticall ###
folder_path = './dataset3'

### INPUT: images to be augmented (Example: choose 5 images for augmentation) ###
selected_image = os.listdir(folder_path + '/train/images')

### INPUT: Define augmentation pipeline ###
train_augment_method = [iaa.GaussianBlur(sigma=(0.0, 2.5)),
                  iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.5, 1.5)), iaa.GammaContrast((1.2, 1.5))]
test_valid_augment_method = [iaa.Fliplr(0.5), iaa.Flipud(0.5)]
# augment_method = [iaa.Flipud(0.5), iaa.GaussianBlur(sigma=(0.0, 2.5)), iaa.Fliplr(0.5)]


# Apply augmentation to generate specified number of images (TRAIN DATASET)
augmented_images = []
for i in range(len(test_valid_augment_method)):
    for image_file in selected_image:

        ### INPUT: Image path to read ###
        image = cv2.imread(folder_path + '/train/images/' + image_file)

        image_height, image_width, _ = image.shape

        ### INPUT: Annotation path to read ###
        bounding_boxes = parse_yolo_annotation(folder_path + "/train/labels/" + image_file.replace(".jpg", ".txt"), image_height, image_width)

        seq = iaa.Sequential([
            # train_augment_method[i]
            test_valid_augment_method[i]
        ])

        augmented_image, augmented_bboxes = seq(image=image, bounding_boxes=bounding_boxes)
        augmented_images.append((augmented_image, augmented_bboxes))

        # save augmented image and annotation

        ### INPUT: Path to save image ###        
        # dest_path = '/train/'
        if i == 0: dest_path = '/test/'
        else: dest_path = '/val/'

        output_image_path = folder_path + dest_path + 'images/' + image_file[:-4] + f'-aug{i}.jpg' # new name: end with '-aug.jpg'

        cv2.imwrite(output_image_path, augmented_image)

        ### INPUT: Path to save annotation ###
        output_annotation_path = folder_path + dest_path + 'labels/' + image_file[:-4] + f'-aug{i}.txt' # new name: end with '-aug.txt'

        with open(output_annotation_path, "w") as file:
            for bbox in augmented_bboxes.bounding_boxes:
                x_center, y_center, width, height = revert_coordinate(int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2), image_height, image_width)
                file.write(str(int(bbox.label)) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(width) + ' ' + str(height) + "\n")

# Apply augmentation to generate specified number of images (TEST & VALID DATASET)
# augmented_images = []
# for i in range(len(test_valid_augment_method)):
#     for image_file in selected_image:

#         ### INPUT: Image path to read ###
#         image = cv2.imread(folder_path + '/validation/images/' + image_file)

#         image_height, image_width, _ = image.shape

#         ### INPUT: Annotation path to read ###
#         bounding_boxes = parse_yolo_annotation(folder_path + "/validation/labels/" + image_file.replace(".jpg", ".txt"), image_height, image_width)

#         seq = iaa.Sequential([
#             test_valid_augment_method[i]
#         ])

#         augmented_image, augmented_bboxes = seq(image=image, bounding_boxes=bounding_boxes)
#         augmented_images.append((augmented_image, augmented_bboxes))

#         # save augmented image and annotation

#         ### INPUT: Path to save image ###        
#         if i == 0: dest_path = '/test/'
#         else: dest_path = '/validation/'

#         output_image_path = folder_path + dest_path + 'images/' + image_file[:-4] + f'-aug{i}.jpg' # new name: end with '-aug.jpg'

#         cv2.imwrite(output_image_path, augmented_image)

#         ### INPUT: Path to save annotation ###
#         output_annotation_path = folder_path + dest_path + 'labels/' + image_file[:-4] + f'-aug{i}.txt' # new name: end with '-aug.txt'

#         with open(output_annotation_path, "w") as file:
#             for bbox in augmented_bboxes.bounding_boxes:
#                 x_center, y_center, width, height = revert_coordinate(int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2), image_height, image_width)
#                 file.write(str(int(bbox.label)) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(width) + ' ' + str(height) + "\n")

# Optional: Visualize augmented images
# for idx, (augmented_image, augmented_bboxes) in enumerate(augmented_images):
#     for bbox in augmented_bboxes.bounding_boxes:
#         cv2.rectangle(augmented_image, (int(bbox.x1), int(bbox.y1)), (int(bbox.x2), int(bbox.y2)), (0, 255, 0), 2)

#     cv2.imshow(f"Augmented Image {idx+1}", augmented_image)
#     cv2.waitKey(0)
#     break

# cv2.destroyAllWindows()