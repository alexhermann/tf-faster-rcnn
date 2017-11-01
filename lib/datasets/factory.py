from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets.pascal_voc import pascal_voc

from .mmsvoc import mmsvoc
from datasets.mmsvoc import mmsvoc
#from datasets.mmsvoctest import mmsvoctest
from datasets.mmsvocval import mmsvocval
import numpy as np




__sets = {}

def _selective_search_IJCV_top_k(split, year, top_k):
    """Return an imdb that uses the top k proposals from the selective search
    IJCV code.
    """
    imdb = pascal_voc(split, year)
    imdb.roidb_handler = imdb.selective_search_IJCV_roidb
    imdb.config['top_k'] = top_k
    return imdb

# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007', '2012', '0712']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year:
                pascal_voc(split, year))


# Set up kittivoc
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'kittivoc_{}'.format(split)
        print(name)
        __sets[name] = (lambda split=split: kittivoc(split))
# Set up mmsvoc
for split in ['train', 'val', 'trainval', 'test']:
    name = 'mmsvoc_{}'.format(split)
    print(name)
    __sets[name] = (lambda split=split: mmsvoc(split))

#for split in ['train', 'val', 'trainval', 'test']:
#    name = 'mmsvoctest_{}'.format(split)
#    print(name)
#    __sets[name] = (lambda split=split: mmsvoctest(split))

for split in ['train', 'val', 'trainval', 'test']:
    name = 'mmsvocval_{}'.format(split)
    print(name)
    __sets[name] = (lambda split=split: mmsvocval(split))
# # KITTI dataset
# for split in ['train', 'val', 'trainval', 'test']:
#     name = 'kitti_{}'.format(split)
#     print name
#     __sets[name] = (lambda split=split: kitti(split))

# Set up coco_2014_<split>
for year in ['2014']:
    for split in ['train', 'val', 'minival', 'valminusminival']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
    for split in ['test', 'test-dev']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# NTHU dataset
for split in ['71', '370']:
    name = 'nthu_{}'.format(split)
    print (name)
    __sets[name] = (lambda split=split: nthu(split))


def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        print (list_imdbs())
        raise KeyError('Unknown datasets: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
