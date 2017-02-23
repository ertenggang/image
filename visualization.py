from utils import get_image_name, get_timestamp
import os

result_root = 'results'

def adapt_srcpath(pathname):
  return pathname.replace('.', '../..',1)

def generate_casetable_image(match_list):
  html = '<table>'
  cnt = 0
  for ml in match_list:
    cnt += 1
    pic_row = '<tr><td>'+str(cnt)+'</td><td><img src="' + adapt_srcpath(ml[0]) + '" width="200" height="200"></td>'
    query_name = get_image_name(ml[0], 'query')
    data_row = '<tr><td></td><td>'+query_name+'</td>'
    for m in ml[1]:
      pic_row += '<td><img src="' + adapt_srcpath(m[0]) + '" width="200" height="200"></td>'
      data_row += '<td>' + str(m[1]) +'</td>'

    pic_row += '</tr>'
    data_row += '</tr>'

    html += pic_row
    html += data_row

  html += '</table>'
  return html


def generate_casetable(match_list):
  result_html = '<table border="1">'
  cnt = 0
  for (session, mlists) in match_list.iteritems():
    session_row = '<tr><td>'+session+'</td><td><table>' 
    for ml in mlists:
      cnt += 1
      pic_row = '<tr><td>'+str(cnt)+'</td><td><img src="' + adapt_srcpath(ml[0]) + '" width="200" height="200"></td>'
      query_name = get_image_name(ml[0], 'query')
      data_row = '<tr><td></td><td>'+query_name+'</td>'
      for m in ml[1]:
        pic_row += '<td><img src="' + adapt_srcpath(m[0]) + '" width="200" height="200"></td>'
        data_row += '<td>' + str(m[1]) +'</td>'

      pic_row += '</tr>'
      data_row += '</tr>'
      session_row += pic_row
      session_row += data_row
    session_row += '</table></td></tr>'

    result_html += session_row

  result_html += '</table>'
  return result_html

def generate_result_table(table_head, data_list):
  html = '<table border="1"><tr><th>&nbsp</th>'
  for th in table_head:
    html += '<th>' + str(th) + '</th>'
  html += '</th>'
  for k in data_list.keys():
    table_row = '<tr>'+ '<th>' + k + '</th>'
    for item in data_list[k]:
      table_row += '<td>' + str(item) +'</td>'
    table_row += '</tr>'
    html += table_row
  html += '</table>'
  return html

def generate_experiment_info(info_dict):
  result_html = '<ul>'
  for (key, value) in info_dict.iteritems():
    tmp_str = None
    if type(value) == list:
      tmp_str = '->'.join(value)
    else:
      tmp_str = value
    result_html += '<li>' + key + ' : ' + tmp_str + '</li>'

  result_html += '</ul>'
  return result_html

def generate_result(match_list, opts=None, flag=None):  
  if flag:
    file_name = flag
  else:
    timestr = get_timestamp()
    file_name = flag+timestr+'.html'
  result_file = os.path.join(result_root, file_name)

  htmlstr = '</!DOCTYPE html><html><head><title>'+timestr+'</title><head><body>'
  if opts:
    htmlstr += generate_experiment_info(opts.info)
  htmlstr += generate_casetable(match_list)
  htmlstr += '</body></html>'

  with open(result_file, 'w') as f:
    f.write(htmlstr)


