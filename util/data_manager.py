from __future__ import print_function, absolute_import
import os
import os.path as osp
import numpy as np
import sys 
sys.path.append("..") 
import glob

from utils import mkdir_if_missing,  write_json, read_json

from IPython import embed
import re


class Market1501(object):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html
    
    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'Market'

    def __init__(self, root = 'data', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()

        # data_dir, ID,CAM ID, number of pictures
        self._process_dir(self.train_dir)
        
        
    def _check_before_run(self):
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))
            
    def _process_dir(self, dir_path, relabel = False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d))')
        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).group())
            if pid == -1: continue
            pid_container.add(pid)
        embed()
        
if __name__ == '__main__':
    data = Market1501(root = '/home/ls')




