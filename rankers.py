from abc import ABCMeta, abstractmethod
import os

import cv2
import numpy as np
import pickle

import feature_extractors
import feature_matchers
from utils import get_query_key

class ranker_base():
  __metaclass = ABCMeta

  @abstractmethod
  def match(self, file_list):
    pass

  def rerank(self, match_list, opts=None):
    new_list = dict()
    for (query_session, mlist) in match_list.items():
      new_list[query_session] = []
      for m in mlist:
        q = m[0]
        score_list = m[1]
        gs = [g[0] for g in score_list]
        mn = self.rank(q, gs, opts)
        m = (q, mn)
        new_list[query_session].append(m)
    return new_list

class local_features_rank(ranker_base):
  def __init__(self):
    self.feature_type = 'SIFT'
    self.match_method = 'RANSAC'
    self.flag = ''

  def rank(self, query, gallery_list, opts):
    feature_extractor = feature_extractors.local_features(opts, self.feature_type)

    query_feas = feature_extractor.feature_extract([query])
    gallery_feas = feature_extractor.feature_extract(gallery_list)

    feature_matcher = feature_matchers.local_features_matcher_v2(opts,self.match_method)
    match_result = feature_matcher.batch_match(query_feas, gallery_feas)
    self.flag = feature_matcher.flag

    id = get_query_key(query)
    return match_result[id][0][1]

