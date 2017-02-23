from abc import ABCMeta, abstractmethod
import os
import math

import pickle
import cv2
import numpy as np

import point_match
import utils
from utils import get_query_key

class batch_features_matcher():
  __metaclass = ABCMeta

  def __init__(self, opts):
    self.data_name = opts.data_name
    self.cache_root = './measure'
    self.k = 5 # top k to remain


  @abstractmethod
  def match(self, f1, f2):
    pass

  def batch_match(self, query_feas, gallery_feas, gallery_name="", query_name="", enable_cache = False):

    match_cache = os.path.join(self.cache_root, gallery_name +'_'+query_name+'_matchscores.pkl')

    if enable_cache == True and os.path.isfile(match_cache):
      with open(match_cache, 'rb') as f:
        print 'cache file: ' + match_cache + ' exists!'
        match_result = pickle.load(f)
    else:
      match_result = dict()
      for (q, qf) in query_feas.items():
        qk = get_query_key(q)
        scores = dict()
        for (g, gf) in gallery_feas.items():
          scores[g] = self.match(qf, gf)

        cmp_method = getattr(utils, 'cmp_'+self.flag)
        top_k = sorted(scores.items(), key=lambda x:x[1], cmp=cmp_method)[:self.k]
    
        if not match_result.has_key(qk):
          match_result[qk] = []
        match_result[qk].append([q, top_k])

        if enable_cache == True:
          if not os.path.exists(self.cache_root):
            os.makedirs(self.cache_root)
          with open(match_cache, 'wb') as f:
            pickle.dump(match_result, f)

    return match_result

class l2(batch_features_matcher):
  def __init__(self, opts):
    batch_features_matcher.__init__(self, opts)
    self.flag = 'distance'
    self.info = 'l2'
    self.cache_root = os.path.join(self.cache_root, self.info, 'cache', self.data_name)

  def match(self, f1, f2):
    d = f1 - f2
    return sum(d*d)

class cosine_simialrity(batch_features_matcher):
  def __init__(self, opts):
    batch_features_matcher.__init__(self, opts)
    self.flag = 'similarity'
    self.info = 'cosine'
    self.cache_root = os.path.join(self.cache_root, self.info, 'cache', self.data_name)

  def match(self, f1, f2):
    s = sum(f1*f2)
    fl = math.sqrt(sum(f1*f1))
    fr = math.sqrt(sum(f2*f2))
    return s/fl/fr

class local_features_matcher(batch_features_matcher):
  def __init__(self, opts, match_method='ransac'):
    batch_features_matcher.__init__(self, opts)
    self.flag = 'similarity'
    self.match_method = match_method
    self.info = 'match_point'
    self.cache_root = os.path.join(self.cache_root, self.info, 'cache', self.data_name)

  def match(self, qfea, gfea):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(qfea['dep'], gfea['dep'])

    transform = point_match.fit_transform()
    min_fit_num = 4
    max_iter = 1000
    threshold = 10
    fit_num_thresh = 4

    data = []
    for m in matches:
      d = np.r_[np.array(qfea['kp'][m.queryIdx]), np.array(gfea['kp'][m.trainIdx])]
      data.append(d)
    data = np.array(data)

    point_matcher = getattr(point_match, self.match_method)
    matches_score = point_matcher(data,transform,min_fit_num, max_iter,threshold,fit_num_thresh,False, False)
    return matches_score


class local_features_matcher_v2(batch_features_matcher):
  def __init__(self, opts, match_method='RANSAC'):
    batch_features_matcher.__init__(self, opts)
    self.flag = 'distance'
    self.match_method = getattr(cv2, match_method)
    self.rej_threshold = 10
    self.max_iter= 1000
    self.confidence = 0.99
    self.info = 'transform_match'
    self.cache_root = os.path.join(self.cache_root, self.info, 'cache', self.data_name)

  def match(self, qfea, gfea):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(qfea['dep'], gfea['dep'])

    srcpoints = []
    dstpoints = []
    for m in matches:
      srcpoints.append(qfea['kp'][m.queryIdx])
      dstpoints.append(gfea['kp'][m.trainIdx])

    srcpoints = np.array(srcpoints)
    dstpoints = np.array(dstpoints)

    transform = cv2.findHomography(srcpoints, dstpoints, self.match_method, self.rej_threshold, None, self.max_iter, self.confidence)[0]
    if transform is None:
      return float("inf")
    else:
      s = qfea['size']
      srcpoints = np.array([[0,0], [0, s[0]], [s[1], 0], [s[1], s[0]]])
      s = gfea['size']
      dstpoints = np.array([[0,0], [0, s[0]], [s[1], 0], [s[1], s[0]]])

      ori_points = np.c_[srcpoints, np.ones(srcpoints.shape[0])].T
      c = np.dot(transform, ori_points)
      c = c.T[:,:-1]
      diff = c-dstpoints
      error = np.mean(np.sqrt(np.sum(diff*diff,1)))
      return error




