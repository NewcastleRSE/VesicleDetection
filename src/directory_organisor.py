import os 

def create_unique_directory_file(path):
    """
        Checks if a directory/file name already exists. If it does, 
        it creates a new directory with (i) appended.
    """
    name, extension = os.path.splitext(path)
    c = 1
    while os.path.exists(path):
        path = name + '_(' + str(c) + ')' + extension 
        c += 1 
    return path 