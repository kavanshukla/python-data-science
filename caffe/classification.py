import numpy as np
import caffe
import sys
import os
import glob

caffe_root = sys.argv[1] + '/'  #Example input path: '/usr/caffe-master/'
images_folder_path = sys.argv[2]    #path in input argument should have '/' at the end, Example input path: '/usr/caffe-master/images/'

sys.path.insert(0, caffe_root + 'python')

if os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
    print 'CaffeNet found.'
else:
    print 'Downloading pre-trained CaffeNet model...'
    os.system('../scripts/download_model_binary.py ../models/bvlc_reference_caffenet')


caffe.set_mode_cpu()
model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

net = caffe.Net(model_def, model_weights, caffe.TEST)
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', mu)
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))

net.blobs['data'].reshape(50, 3, 227, 227)

labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
if not os.path.exists(labels_file):
    os.system('../data/ilsvrc12/get_ilsvrc_aux.sh')
    
labels = np.loadtxt(labels_file, str, delimiter='\t')

limit = 0
print('image_name' +'        '+'label')

#Displaying image_name and its predicted label

for image_path in glob.iglob(images_folder_path+'*.jpg'):
	if(limit<10):
		image = caffe.io.load_image(image_path)
		transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image
        output = net.forward()
        output_prob = output['prob'][0]
        prediction = output_prob.argmax()

        lb = labels[prediction]

        image_name = image_path.split(images_folder_path)[1]

        result_str = image_name +'    '+lb
        print(result_str)
		limit = limit + 1
