# coding=utf-8
import os 

import pickle

import visualization
from utils import get_image_name, get_timestamp
import utils


class image_evaluator():
  def __init__(self):
    self.case_list = []
    self.score_type = ""
    self.result_root = 'results'
    self.badcases = dict()
    self.result = dict()
    self.total_num = 0
    self.positive_num = 0
    self.negative_num = 0
    self.max_score = float('-inf')
    self.min_score = float('inf')


  def add_case(self, match_result):
    for (key, value_list) in match_result.items():
      for q_match in value_list:
        match_score = q_match[1][0][1]
        if match_score > self.max_score:
          self.max_score = match_score
        if match_score < self.min_score:
          self.min_score = match_score
        query_id = get_image_name(q_match[0], 'query')
        gallery_id = get_image_name(q_match[1][0][0], 'gallery')
        self.total_num += 1
        if query_id == 'other':
          pn_flag = 0
          self.negative_num += 1
        else:
          pn_flag = 1
          self.positive_num += 1

        if  query_id == gallery_id:
          correct = 1
        else:
          correct = 0
        self.case_list.append((match_score, pn_flag, correct, q_match))

  def guess_thresholds(self):
    num = 10
    binsize = (self.max_score - self.min_score)/num
    thresholds = []
    i = self.min_score
    while i < self.max_score:
      thresholds.append(i)
      i += binsize
    return thresholds

  def run(self, thresholds = None, show_badcase = False):
    if thresholds is None:
      thresholds = self.guess_thresholds()
    self.case_list = sorted(self.case_list, key=lambda x:x[0], cmp=getattr(utils, 'cmp_'+self.score_type))
    [scores, tn, correct, info] = zip(*self.case_list)
    false_positives = []
    false_negatives = []
    accus = []
    self.thresholds = thresholds

    for threshold in thresholds:
      i = 0
      cmper = getattr(utils, 'cmp_'+self.score_type)
      while i < self.total_num and cmper(threshold, self.case_list[i][0]) >= 0:
        i += 1

      false_positive_num = (i - sum(tn[0:i]))
      false_negative_num = sum(tn[i:])

      correct_num = sum(correct[:i])

      if self.negative_num == 0:
        false_positives.append(0)
      else:
        false_positives.append(float(false_positive_num)/self.negative_num)
      
      if self.positive_num == 0:
        false_negatives.append(0)
      else:
        false_negatives.append(float(false_negative_num)/self.positive_num)

      base = sum(tn[0:i])
      if base == 0:
        accus.append(0)
      else:
        accus.append(float(correct_num)/base)

      if show_badcase:
        self.show_badcase = True
        id = str(threshold)
        self.badcases[id] = dict()
        self.badcases[id]['false positive'] = [k[3] for k in self.case_list[:i] if k[1]==0]
        self.badcases[id]['false negative'] = [k[3] for k in self.case_list[i:] if k[1]==1]
        self.badcases[id]['classification error'] = [k[3] for k in self.case_list[:i] if k[1]==1 and k[2]==0]

    self.result['false_positives'] = false_positives
    self.result['false_negatives'] = false_negatives
    self.result['accus'] = accus
    return self.result

  def generate_report(self, opts):
    timestr = get_timestamp()
    filedir = os.path.join(self.result_root, timestr)
    os.makedirs(filedir)
    
    overviewfile = os.path.join(filedir, 'index.html')
    html = "<h2>Test settings:</h2>"
    html += visualization.generate_experiment_info(opts.info)
    html += '<h2> Test result:</h2>'
    html += '<p>Test ' + str(self.total_num) + ' query: ' + str(self.negative_num) + ' negative queries and '+str(self.positive_num) + ' positive queries.</p>'
    html += visualization.generate_result_table(self.thresholds, self.result)
    if self.show_badcase:
      html += '<h2> Badcase links:</h2>'
      html += '<ul>'
      for th in self.thresholds: 
        html += '<li><a href = "' + str(th)+'.html"> ' +str(th) + '</a></li>'
      html += '</ul>'

    [a,b,c, case_list] = zip(*self.case_list)
    html += visualization.generate_casetable_image(case_list)

    html = '</!DOCTYPE html><html><head><title>overview</title><head><body>'+html+'</body></html>'
    with open(overviewfile, 'w') as f:
      f.write(html)

    if self.show_badcase:
      for th in self.thresholds:
        file = os.path.join(filedir, str(th)+'.html')
        html = visualization.generate_casetable(self.badcases[str(th)])
        html = '</!DOCTYPE html><html><head><title>overview</title><head><body>'+html+'</body></html>'
        with open(file, 'w') as f:
          f.write(html)












def evaluate_negative(match_result, thresholds):
  match_scores = []
  for (key, value_list) in match_result.items():
    for q_match in value_list:
      match_scores.append(q_match[1][0][1])

  match_scores = sorted(match_scores)
  total_num = len(match_scores)

  false_positives = []
  for threshold in thresholds:
    false_positive = 0
    i = 0
    while i < total_num and match_scores[i] < threshold :
      false_positive += 1
      i += 1
    false_positives.append(false_positive)

  return total_num, false_positives
  

def evaluate_positive(match_result, thresholds, dumps_badcases = False):
  match_scores = []
  truths = []
  cnt = 0
  for key in sorted(match_result.keys()):
    value_list = match_result[key]
    for q_match in value_list:
      cnt += 1
      match_scores.append(q_match[1][0][1])
      query_id = get_image_name(q_match[0], 'query')
      gallery_id = get_image_name(q_match[1][0][0], 'gallery')
      if query_id == gallery_id:
        truths.append(1)
      else:
        truths.append(0)

  match_scores = zip(match_scores, truths)
  match_scores = sorted(match_scores, key=lambda x:x[0], reverse=True)

  total_num = len(match_scores)

  false_negatives = []
  accus = []
  for threshold in thresholds:
    false_negative = total_num
    i = 0
    while i < total_num and match_scores[i][0] > threshold:
      false_negative -= 1
      i += 1

    false_negatives.append(false_negative)
    if(i == 0): accu = 0
    else :accu = float(sum(zip(*match_scores)[1][0:i]))/i
    accus.append(accu)

  return total_num, false_negatives, accus

