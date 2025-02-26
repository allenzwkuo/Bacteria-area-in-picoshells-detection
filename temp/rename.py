import os
from PIL import Image

# Path where your PNG mask images are stored
task_folder = '.'
# Path where you want the renamed masks (as JPG)
export_folder = '../PicoshellDataset_UNET/test_masks'

# Create black 100x100 mask
def create_black_mask(new_filename):
    """Creates and saves a black mask image of 100x100"""
    black_mask = Image.new("RGB", (100, 100), (0, 0, 0))  # black image
    black_mask.save(os.path.join(export_folder, new_filename))
    print(f"Black mask created for {new_filename}")

# Get a list of all PNG files, considering the task number (e.g., task-5.png)
image_files = [f for f in os.listdir(task_folder) if f.endswith('.png')]

# Extract task numbers from the filenames (e.g., task-5-annotation-5-by-1-tag-bacteria-0.png -> 5)
task_numbers = sorted([int(f.split('-')[1]) for f in image_files if f.startswith('task-')])

# Find missing task numbers
existing_tasks = set(task_numbers)
start_task = task_numbers[0]
end_task = task_numbers[-1]

# Now rename existing and create new black masks for missing tasks
counter = 1
for task_num in range(start_task, end_task + 1):
    new_filename = f"I{counter:02d}_mask.jpg"
    
    # Check if the task exists
    matching_files = [f for f in image_files if f.startswith(f'task-{task_num}-')]
    if matching_files:
        # If the task exists, rename the corresponding task-*.png to Axx_mask.jpg
        old_filename = matching_files[0]
        old_filepath = os.path.join(task_folder, old_filename)
        new_filepath = os.path.join(export_folder, new_filename)
        
        # Convert PNG to JPG and save
        img = Image.open(old_filepath)
        img = img.convert('RGB')
        img.save(new_filepath)
        print(f"Renamed and saved: {new_filename}")
    else:
        # If the task is missing, create a black mask
        create_black_mask(new_filename)
    
    # Increment the counter for the next mask
    counter += 1

print("All tasks processed.")
