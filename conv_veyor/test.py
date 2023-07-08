import os
import re
for case in ['train','test']:
    files = os.listdir(f'./test_images/{case}')
    with open(f'./test_images/rec_{case}.txt') as f:
        index = f.readlines()
    files = [re.search(r'(?<=Word_)\d+(?=_)', x).group() for x in files]
    index = [re.search(r'(?<=Word_)\d+(?=_)', x).group() for x in index]
    files = sorted([int(x) for x in files])
    index = sorted([int(x) for x in index])
    assert set(index).symmetric_difference(set(files)) == set()
