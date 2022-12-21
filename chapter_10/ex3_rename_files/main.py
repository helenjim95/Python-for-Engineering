import os
import shutil


def rename_imgs(folder, output_folder):
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".JPEG") or filename.endswith(
                ".JPG"):
            new_filename = filename.replace(".jpeg", ".jpg").replace(".JPG", ".jpg").replace(".JPEG", ".jpg")
            shutil.copy(os.path.join(folder, filename), os.path.join(output_folder, new_filename))
    # print('All Files Renamed')
    # print('New Names are')
    # res = os.listdir(output_folder)
    # print(res)


def main():
    input_folder = "input"
    output_folder = "output"
    rename_imgs(input_folder, output_folder)


if __name__ == "__main__":
    main()
