"""
Create files labelled_data.data and classes.names

Algorithm:
    Set up full paths --> Read file classes.txt -->
    --> Create file classes.names -->
    --> Create file labelled_data.data

Return:
    classes.names, labelled_data.data
"""

full_path_to_images = r'C:\Users\15734\PycharmProjects\nike_series\data\images\Nike SB Dunk Low StrangeLove'


# create class names
c = 0
with open(full_path_to_images + '/' + 'classes.names', 'w') as names, \
     open(full_path_to_images + '/' + 'classes.txt', 'r') as txt:
    for class_i in txt:
        print(class_i)
        names.write(class_i)
        c += 1


with open(full_path_to_images + '/' + 'labelled_data.data', 'w') as data:
    data.write('classes = ' + str(c) + '\n')

    # train.txt and test.txt paths
    data.write('train = ' + full_path_to_images + '/' + 'train.txt' + '\n')
    data.write('valid = ' + full_path_to_images + '/' + 'test.txt' + '\n')

    # class names path
    data.write('names = ' + full_path_to_images + '/' + 'classes.names' + '\n')

    # weights
    data.write('backup = backup')
