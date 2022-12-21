import glob
import os
import shutil
import magic
import filetype

def collect_files(folder, outfolder, ext="pdf", use_magic=False):
    print("Collecting files")
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    for root, dirs, files in os.walk(folder, topdown=False):
        for file_ in files:
            if file_.endswith(ext):
                shutil.copy(os.path.join(root, file_), os.path.join(outfolder, file_))
            else:
                print("File does not end with extension: " + file_)
                if use_magic:
                    fileinfo = filetype.guess(file_)
                    mime_type = fileinfo.mime
                    if mime_type is None:
                        print('Cannot guess file type!')
                    else:
                        if mime_type == ext:
                            print('File extension: %s' % mime_type)
                            magic_ext = " ".join(glob.glob(folder + "/*." + ext))
                            magic_ext = magic_ext.split()
                            for i in magic_ext:
                                os.system("cp " + i + " " + outfolder)
                    # try:
                    #     shutil.copy(os.path.join(root, file_), os.path.join(outfolder, file_))
                    # except:
                    #     pass
    print("Done")


def main():
    folder = "input"
    outfolder = "output"
    print("main")
    # collect_files(folder, outfolder, ext="pdf", use_magic=True)
    collect_files(folder, outfolder, ext="application/pdf", use_magic=True)


if __name__ == "__main__":
    main()
