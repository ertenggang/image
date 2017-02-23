# coding=utf-8
from abc import ABCMeta, abstractmethod
import glob
import os
import sys

import pickle
import numpy as np


import feature_extractors
import feature_matchers
import rankers
import evaluators
import visualization

from utils import get_file_list


class settings():
  def __init__(self):
    # testing pipeline
    self.pipeline = { 'feature_extractor':{'stage':0,'name':'tfnet', 'enable_cache':True},
                      'feature_matcher':{'stage':1, 'name':'cosine_simialrity', 'enable_cache':False},
                      # 'ranker':{'stage':2, 'name':'local_features_rank'}
    }

    # data path
    self.data_root = './data/CNT China-December issue 2016'
    self.data_name = os.path.split(self.data_root)[-1]
    self.query_negative = 'query_index/other'
    self.query_positive = 'query_index/positive'
    self.query_positive1 = 'query_index/positive/CNT China-December issue 2016_页面_001_图像_0001'
    self.query_positive2 = 'query_index/positive/CNT China-December issue 2016_页面_002_图像_0004'
    self.gallery = 'gallery'

    self.info = { 'data': self.data_name,
                  'gallery' : self.gallery,
                  'query': None,
                  'pipeline': self.get_pipeline()
    }
    self.thresholds = None

  def get_pipeline(self):
    l = len(self.pipeline)
    p = ['']*l
    for (k, v) in self.pipeline.items():
      stage = v['stage']
      name = v['name']
      p[stage] = name
    return p

class test():
  __metaclass = ABCMeta
  def __init__(self, query_to_test = [ 'query_positive']):
    self.query_to_test = query_to_test
    self.evaluator = evaluators.image_evaluator()
    self.opts = settings()

    self.opts.info['query'] = []
    for q in query_to_test:
      self.opts.info['query'].append(getattr(self.opts, q))
    self.opts.info['query'] = str(self.opts.info['query'])

  def run(self):
    enable_cache = True
    if enable_cache:
      cachefile = 'test_cache.pkl'
      if os.path.isfile(cachefile):
        with open(cachefile, 'rb') as f:
          self.evaluator = pickle.load(f)
      else:
        for query_flag in self.query_to_test:
          [match_result, self.evaluator.score_type] = self.run_on_query(query_flag)
          self.evaluator.add_case(match_result)
        with open(cachefile, 'wb') as f:
          pickle.dump(self.evaluator, f)
    
    self.evaluator.run(self.opts.thresholds, True)
    self.evaluator.generate_report(self.opts)

  @abstractmethod
  def run_on_query(self, attr_query):
    pass

class test_type1(test):
  def __init__(self, query_to_test = [ 'query_positive']):
    test.__init__(self, query_to_test)

  def run_on_query(self, attr_query):
    score_type = 'distance'

    #================
    # extract features
    feature_extractor = getattr(feature_extractors, self.opts.pipeline['feature_extractor']['name'])(self.opts)
    gallery_feas = feature_extractor.batch_features_extract(self.opts, 'gallery', self.opts.pipeline['feature_extractor']['enable_cache'])
    query_feas = feature_extractor.batch_features_extract(self.opts, attr_query, self.opts.pipeline['feature_extractor']['enable_cache'])

    #==============
    # feature select if needed

    #================
    # match
    matcher = getattr(feature_matchers, self.opts.pipeline['feature_matcher']['name'])(self.opts)
    score_type = matcher.flag
    match_result = matcher.batch_match(query_feas, gallery_feas, enable_cache=self.opts.pipeline['feature_matcher']['enable_cache'])

    #================
    # re-rank if needed
    if self.opts.pipeline.has_key('ranker'):
      ranker = getattr(rankers, self.opts.pipeline['ranker']['name'])()
      match_result = ranker.rerank(match_result, self.opts)
      score_type = ranker.flag


    return match_result, score_type

    # visualization.generate_result(match_result, opts)
  #   thresholds = [0, 10 , 20, 30, 40]
  # , false_negatives, accus = evaluators.evaluate_positive(match_result, thresholds)

  #   print('total number of positive:', total_num)
  #   for i in range(len(thresholds)):
  #     print('===================================')
  #     print('threshold:', thresholds[i])
  #     print('fasle negative rate: ' , float(false_negatives[i])/ total_num)
  #     print('accuracy: ', accus[i])


if __name__ == '__main__':
  session = test_type1([ 'query_positive', 'query_negative'])
  # session = test_type1([ 'query_opsitive1'])
  session.run()


# if __name__ == '__main__':
#   opts = settings()


#   #============================================
#   # testing googlenet + l2
#   processor = getattr(feature_extractors, opts.feature_type)(opts)

#   print("=====================================")
#   print("testing positive samples ...")

#   re_match = True
#   match_cache = './match_result.pkl'
#   if re_match == False and os.path.isfile(match_cache):
#     with open(match_cache) as f:
#       new_list = pickle.load(f)
#   else:

#     match_result = processor.batch_match(opts, 'query_positive', enable_cache=True)

#     ranker = getattr(rankers, opts.rank)(opts)

#     new_list = ranker.rerank(match_result)
    

#     with open('./match_result.pkl', 'wb') as f:
#       pickle.dump(new_list, f)

#   visualization.generate_result(new_list, opts)

#   thresholds = [0, 10 , 20, 30, 40]
#   total_num, false_negatives, accus = evaluators.evaluate_positive(new_list, thresholds)

#   print('total number of positive:', total_num)
#   for i in range(len(thresholds)):
#     print('===================================')
#     print('threshold:', thresholds[i])
#     print('fasle negative rate: ' , float(false_negatives[i])/ total_num)
#     print('accuracy: ', accus[i])

  # print("=====================================")
  # print("testing negative samples ...")
  # opts.query = opts.query_negative
  # query_feas = feature_extraction(opts, 'query')

  # match_cache = 'negative_match_scores.pkl'
  # if os.path.isfile(match_cache):
  #   with open(match_cache, 'rb') as f:
  #     match_result = pickle.load(f)
  # else:
  #   match_result = feature_match(query_feas, gallery_feas, opts)
  #   with open(match_cache, 'wb') as f:
  #     pickle.dump(match_result, f)


  # thresholds = [0 ,50, 80, 100, 200, 300, 400, 500, 600]
  # total_num, false_positives = evaluate_negative(match_result, thresholds)

  # print('total number of negative:', total_num)
  # for i in range(len(thresholds)):
  #   print('===================================')
  #   print('threshold:', thresholds[i])
  #   print('fasle positive rate: ' , float(false_positives[i])/ total_num)


  # print("==================================")
  # print("testing positve samples ...")
  # opts.query = opts.query_positive
  # query_feas = feature_extraction(opts, 'query')

  # match_cache = 'positive_match_scores.pkl'
  # if os.path.isfile(match_cache):
  #   with open(match_cache, 'rb') as f:
  #     match_result = pickle.load(f)
  # else:
  #   match_result = feature_match(query_feas, gallery_feas, opts)
  #   with open(match_cache, 'wb') as f:
  #     pickle.dump(match_result, f)


  # thresholds = [0 ,50, 80, 100, 200, 300, 400, 500, 600]
  # total_num, false_negatives, accus = evaluate_positive(match_result, thresholds)

  # print('total number of negative:', total_num)
  # for i in range(len(thresholds)):
  #   print('===================================')
  #   print('threshold:', thresholds[i])
  #   print('fasle negative rate: ' , float(false_negatives[i])/ total_num)
  #   print('accu: ' , accus[i])

  # ======================
  # testing get_file_list
  # ======================
  # input_dir = sys.argv[1]
  # file_list = get_file_list(input_dir)
  # print(file_list )


  # =======================
  # testing feature_extraction
  # =======================
  # opts = settings()
  # feas = feature_extraction(opts)
  # print feas
  
  # =======================
  # testing get_query_label
  # =======================
  # q = './image_match/data/CNT China-December issue 2016/query_index/CNT China-Decemberx/1/dd.jpg'
  # key = get_query_key(q)
  # print (key)