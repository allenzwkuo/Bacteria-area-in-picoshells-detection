import os

folder_path = "."

files = os.listdir(folder_path)

jpg_files = [f for f in files if f.lower().endswith('.jpg')]

jpg_files.sort()

for i, file in enumerate(jpg_files):
    new_name = chr(65 + i) + ".jpg"  
    old_file_path = os.path.join(folder_path, file)
    new_file_path = os.path.join(folder_path, new_name)
    
    os.rename(old_file_path, new_file_path)
    print(f"Renamed: {file} -> {new_name}")
