from abc import ABCMeta, abstractmethod
import os

import pickle
import tensorflow as tf
import cv2

from utils import get_query_key, get_file_list


class batch_features_extractor():
  __metaclass = ABCMeta

  def __init__(self, opts):
    self.cache_root = './feature'
    self.feature_info = 'batch_features'
    self.data_name = opts.data_name
    self.k = 5


  def batch_features_extract(self, opts, flag='query', enable_cache=True):
    print("===================================")
    fea = None
  
    fea_smps = getattr(opts, flag)
    cache_file = os.path.join(self.cache_root, self.feature_info + '_' + fea_smps.replace('/','_') +'.pkl')
    if enable_cache and os.path.isfile(cache_file):
      print('Feature cache file exists : '+ cache_file)
      print('loading '+flag+' features...')
      with open(cache_file, 'rb') as f:
        fea = pickle.load(f)
      print('Loading '+flag+'features completed.')
    else:
      if enable_cache:
        print('Feature cache file not found : '+ cache_file)
      file_dir = os.path.join(opts.data_root, fea_smps)
      file_list = get_file_list(file_dir)
    
      print('Extracting '+flag+' features...')
      fea = self.feature_extract(file_list)
      print('Extracting '+flag+'features completed.')

      if enable_cache:
        print('Saving ' + flag + 'features...')
        if not os.path.exists(self.cache_root):
          os.makedirs(self.cache_root)
        with open(cache_file, 'wb') as f:
          pickle.dump(fea, f)
        print('Saving ' + flag + ' features completed.')
    return fea

  @abstractmethod
  def feature_extract(self, file_list):
    pass


class tfnet(batch_features_extractor):
  def __init__(self, opts):
    batch_features_extractor.__init__(self, opts)
    self.feature_dir = os.path.join(self.cache_root,'tfnet')
    self.cache_root = os.path.join(self.feature_dir, 'cache', self.data_name)
    self.opts = dict()
    self.opts['model_file'] = os.path.join(self.feature_dir, 'classify_image_graph_def.pb')
    self.opts['model_name'] = os.path.splitext(os.path.split(self.opts['model_file'])[-1])[0]
    self.opts['layer_name'] = 'pool_3:0'
    self.feature_info = self.opts['model_name'] + '_' + self.opts['layer_name']

    self.create_graph()
    
  def create_graph(self):
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    model_file = self.opts['model_file']
    print ('creating graph : '+ model_file)
    with tf.gfile.FastGFile(model_file, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      _ = tf.import_graph_def(graph_def, name='')

  def feature_extract(self, file_list):
    # input:
    #   -file_list :A list of pathnames of images. 
    # output:
    #   -fea : A dict of features. Each item has an image's pathname as its key and the image's feature as its value.
    
    fea = dict()
    for image in file_list:
      image_data = tf.gfile.FastGFile(image, 'rb').read()
      
      with tf.Session() as sess:
        # Some useful tensors:
        # 'softmax:0': A tensor containing the normalized prediction across
        #   1000 labels.
        # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
        #   float description of the image.
        # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
        #   encoding of the image.
        # Runs the softmax tensor by feeding the image_data as input to the graph.
        # softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
        fea_tensor = sess.graph.get_tensor_by_name(self.opts['layer_name'])
        features = sess.run(fea_tensor,
                           {'DecodeJpeg/contents:0': image_data})

        fea[image] = features[0][0][0]

      print(image)
    return fea

class local_features(batch_features_extractor):
  def __init__(self, opts, feature_type='SIFT'):
    batch_features_extractor.__init__(self, opts)
    self.feature_type = feature_type
    self.feature_dir = os.path.join(self.cache_root, self.feature_type)
    self.cache_root = os.path.join(self.feature_dir, 'cache')

    self.max_edge = 500
    self.extractor = getattr(cv2.xfeatures2d, self.feature_type+'_create')()

  def feature_extract_(self, image):
    im = cv2.imread(image)
    height, width = im.shape[:2]
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    max_edge = max(height, width)
    scale = float(self.max_edge)/max_edge

    if scale < 1:
      im = cv2.resize(im,(int(scale*width), int(scale*height)))
    kp, dep = self.extractor.detectAndCompute(im, None)

    kp_positions = []
    for k in kp:
      kp_p = k.pt
      kp_positions.append(kp_p)


    fea = dict({'kp':kp_positions, 'dep':dep , 'size':im.shape[:2]})
    return fea

  def feature_extract(self, file_list):
    feas = dict()
    for image in file_list:
      fea = self.feature_extract_(image)
      print image
      print 'number of keypoints: ' + str(len(fea['kp']))
      print("========completed==============")
      feas[image] = fea
    return feas
     
