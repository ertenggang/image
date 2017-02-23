import os
import glob
import time

def get_file_list(root_dir):
  if not os.path.isdir(root_dir):
    print(root_dir + ' is not a directory!')

  image_exts = ["JPG",'jpg']
  file_list = []
  for ext in image_exts: 
    file_list += glob.glob(os.path.join(root_dir, '*.'+ext))

  dflist = os.listdir(root_dir)
  for f in dflist:
    subdir = os.path.join(root_dir, f)
    if os.path.isdir(subdir):
      file_list += get_file_list(subdir)

  return file_list

def get_image_name(image_pathname, flag):
  if flag == 'gallery':
    filename = os.path.split(image_pathname)[1]
    filename = os.path.splitext(filename)[0]
  elif flag == 'query':
    filename = image_pathname.split(os.path.sep)[-3]
  else:
    pass

  return filename


def get_query_key(q):
  # input : 
  #   -q: pathname of a query
  pathlist = q.split(os.path.sep)
  return '_'.join(pathlist[-3:-1])

def get_timestamp():
  t = time.localtime(time.time())
  tstr = time.strftime('%Y-%m-%d-%H-%M-%S', t)
  return tstr


def cmp_distance(x, y):
  if x > y:
    return 1
  elif x < y:
    return -1
  else:
    return 0

def cmp_similarity(x, y):
  if y > x:
    return 1
  elif y < x:
    return -1
  else:
    return 0