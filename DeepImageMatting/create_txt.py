import os
dir_name = 'data/test_input'
# Get list of all files in a given directory sorted by name

with open("test_names.txt", "w") as a:
    files = sorted( filter( lambda x: os.path.isfile(os.path.join(dir_name, x)),
                        os.listdir(dir_name) ) )
    for filename in files:
        f = os.path.join(filename)
        a.write(str(f) + os.linesep) 