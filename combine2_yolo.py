
import os
import shutil

yolotxt_din_path = './TXT Aliff'
yolotxt_aliff_path = './TXT Din'
# yolotxt_amin_path = './TXT Amin'

# Specify the path of the new folder
new_folder_path = './dataset4'
combined_label_folder_path = new_folder_path + '/labels'
combined_image_folder_path = new_folder_path + '/images'

# Create the folder
os.makedirs(new_folder_path, exist_ok=True)
os.makedirs(combined_label_folder_path, exist_ok=True)
os.makedirs(combined_image_folder_path, exist_ok=True)

# Move all files (images and annotations) into 1 folder

def move_files(source_folder, destination_folder):

    label_folder = os.path.join(source_folder, 'labels')
    label_dest_folder = os.path.join(destination_folder, 'labels')

    # Iterate through files in the source folder
    for filename in os.listdir(label_folder):

        # Construct full paths for source and destination files
        source_file = os.path.join(label_folder, filename)
        destination_file = os.path.join(label_dest_folder, filename)

        # Move the file
        shutil.move(source_file, destination_file)

    image_folder = os.path.join(source_folder, 'images')
    image_dest_folder = os.path.join(destination_folder, 'images')

    # Iterate through files in the source folder
    for filename in os.listdir(image_folder):

        # Construct full paths for source and destination files
        source_file = os.path.join(image_folder, filename)
        destination_file = os.path.join(image_dest_folder, filename)

        # Move the file
        shutil.move(source_file, destination_file)


move_files(yolotxt_din_path, new_folder_path)
move_files(yolotxt_aliff_path, new_folder_path)
# move_files(yolotxt_amin_path, new_folder_path)

shutil.move(os.path.join(yolotxt_din_path, 'classes.txt'), new_folder_path)
shutil.move(os.path.join(yolotxt_din_path, 'notes.json'), new_folder_path)