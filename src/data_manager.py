import os


def get_files(file_path):
    files_list = os.listdir(path=file_path)
    files = {}
    for file in files_list:
        file_name = os.path.basename(file).split(".")[0].replace("_", " ")
        files[file_name] = file_path + file
    return files


if __name__ == "__main__":
    files = get_files("./data/references/art_styles/")
    print(files)
