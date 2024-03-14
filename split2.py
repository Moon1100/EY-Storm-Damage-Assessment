import os
import random
import shutil

### INPUT: Specify the dataset path, train-test images will be put in train/test folder inside dataset path ###
dataset_path = "./dataset4"
label_path = os.path.join(dataset_path, 'labels')

train_dataset_path = os.path.join(dataset_path, 'train')
train_label_path = os.path.join(train_dataset_path, 'labels')
train_image_path = os.path.join(train_dataset_path, 'images')

test_dataset_path = os.path.join(dataset_path, 'test')
test_label_path = os.path.join(test_dataset_path, 'labels')
test_image_path = os.path.join(test_dataset_path, 'images')

val_dataset_path = os.path.join(dataset_path, 'val')
val_label_path = os.path.join(val_dataset_path, 'labels')
val_image_path = os.path.join(val_dataset_path, 'images')

# Create the folders
for path in [train_dataset_path, train_label_path, train_image_path,
             test_dataset_path, test_label_path, test_image_path,
             val_dataset_path, val_label_path, val_image_path]:
    os.makedirs(path, exist_ok=True)

train_annotations = []
test_annotations = []
val_annotations = []

class_counts = [0, 0, 0, 0]

# Define the minimum count required for each class in the train dataset
min_count_per_class = 100

# Track the number of annotations added to the train, validation, and test datasets for each class
train_annotations_count = [0, 0, 0, 0]
test_annotations_count = [0, 0, 0, 0]
val_annotations_count = [0, 0, 0, 0]

# Calculate the minimum number of annotations needed for each class
min_annotations_needed = [max(0, min_count_per_class - count) for count in train_annotations_count]

# Priority score dictionary to track the priority score for each annotation file
priority_scores = {}

files = os.listdir(label_path)

# Shuffle the list of files randomly
random.seed(42)
random.shuffle(files)

# Iterate over the annotation files
for filename in files:
    if filename.endswith('.txt'):
        file_path = os.path.join(label_path, filename)

        # Initialize priority score for the current file
        priority_score = 0

        # Open the annotation file
        with open(file_path, 'r') as file:
            for line in file:
                class_index = int(line[0])
                class_counts[class_index] += 1

                # Update priority score based on the current class and minimum annotations needed
                priority_score += min_annotations_needed[class_index]

        # Update priority score dictionary with the priority score for the current file
        priority_scores[filename] = priority_score

# Sort annotation files based on priority scores in ascending order
sorted_files = sorted(priority_scores.items(), key=lambda x: x[1])

# Add files with the lowest priority scores to the train, validation, and test datasets
for filename, _ in sorted_files:
    # Check if the minimum count requirement is satisfied for all classes
    if all(count >= min_count_per_class for count in train_annotations_count):
        break  # Stop iterating if the requirement is met

    # Add the current file to the train dataset
    train_annotations.append(filename)

    # Update train_annotations_count based on the annotations in the current file
    with open(os.path.join(label_path, filename), 'r') as file:
        for line in file:
            class_index = int(line[0])
            train_annotations_count[class_index] += 1

# Add remaining files to the test dataset
test_annotations = [filename for filename in os.listdir(label_path) if filename not in train_annotations]

# Split test annotations for validation and test datasets
random.shuffle(test_annotations)
val_annotations = test_annotations[:len(test_annotations)//2]
test_annotations = test_annotations[len(test_annotations)//2:]

print('--- Train dataset ---')
print("Number of annotations for each class:", train_annotations_count)
print("Number of annotations in train dataset:", len(train_annotations))

print('\n--- Validation dataset ---')
print("Number of annotations for each class:", val_annotations_count)
print("Number of annotations in validation dataset:", len(val_annotations))

print('\n--- Test dataset ---')
print("Number of annotations for each class:", test_annotations_count)
print("Number of annotations in test dataset:", len(test_annotations))

def move_files(source_folder, train_folder, test_folder, val_folder):
    label_folder = os.path.join(source_folder, 'labels')
    image_folder = os.path.join(source_folder, 'images')

    train_label_folder = os.path.join(train_folder, 'labels')
    train_image_folder = os.path.join(train_folder, 'images')

    test_label_folder = os.path.join(test_folder, 'labels')
    test_image_folder = os.path.join(test_folder, 'images')

    val_label_folder = os.path.join(val_folder, 'labels')
    val_image_folder = os.path.join(val_folder, 'images')

    for filename in train_annotations:
        # move training label
        source_label_file = os.path.join(label_folder, filename)
        destination_label_file = os.path.join(train_label_folder, filename)
        shutil.move(source_label_file, destination_label_file)

        # move training image
        source_image_file = os.path.join(image_folder, filename.replace(".txt", ".jpg"))
        destination_image_file = os.path.join(train_image_folder, filename.replace(".txt", ".jpg"))
        shutil.move(source_image_file, destination_image_file)

    for filename in val_annotations:
        # move validation label
        source_label_file = os.path.join(label_folder, filename)
        destination_label_file = os.path.join(val_label_folder, filename)
        shutil.move(source_label_file, destination_label_file)

        # move validation image
        source_image_file = os.path.join(image_folder, filename.replace(".txt", ".jpg"))
        destination_image_file = os.path.join(val_image_folder, filename.replace(".txt", ".jpg"))
        shutil.move(source_image_file, destination_image_file)

    for filename in test_annotations:
        # move test label
        source_label_file = os.path.join(label_folder, filename)
        destination_label_file = os.path.join(test_label_folder, filename)
        shutil.move(source_label_file, destination_label_file)

        # move test image
        source_image_file = os.path.join(image_folder, filename.replace(".txt", ".jpg"))
        destination_image_file = os.path.join(test_image_folder, filename.replace(".txt", ".jpg"))
        shutil.move(source_image_file, destination_image_file)

move_files(dataset_path, train_dataset_path, test_dataset_path, val_dataset_path)
