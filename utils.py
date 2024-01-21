def update_type(filename, new_contents):
    with open(filename, 'w') as file:
        file.write(new_contents)

def get_type(filename):
    with open(filename, 'r') as file:
        file_contents = file.read()
    return file_contents