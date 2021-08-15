import os

def get_file_name(file_path, with_extension=False):
    file_path = file_path.replace('\\', '/')
    [file_name, ext] = file_path.split('/')[-1].split('.')
    if with_extension:
        return file_name + '.' + ext
    else:
        return file_name

def gen_dir(new_dir, remove_old=False):
    import os, shutil
    if remove_old:
        if os.path.exists(new_dir):
            for file_name in os.listdir(new_dir):
                file_path = os.path.join(new_dir, file_name)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(e)
        else:
            os.mkdir(new_dir)


    elif not os.path.exists(new_dir):
        os.mkdir(new_dir)

    return (new_dir + '/')