import csv
import urllib2
from multiprocessing.dummy import Pool

def process_url((url, cnt, total, name)):
    img = urllib2.urlopen(url)
    with open('images/'+name, 'wb') as op:
        op.write(img.read())
    print '[', cnt, '/', total, '] ', name

base_url = 'http://media.findartinfo.com/images/artwork/'
f = open('final_input_with_image_urls.csv', 'rb')
reader = csv.reader(f)
reader.next()
lst = list(reader)
cnt = 1
total = len(lst)
urls = []
for i in range(len(lst)):
    if lst[i][-1]:
        year, month, name = lst[i][-1].split('_')
        urls.append((base_url+year+'/'+month+'/'+name, cnt, total, name))
        cnt += 1
    
pool = Pool(processes=10)
pool.map(process_url, urls)
