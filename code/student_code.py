import cv2
import numpy as np
import pickle
from utils import load_image, load_image_gray
import cyvlfeat as vlfeat
import sklearn.metrics.pairwise as sklearn_pairwise
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from IPython.core.debugger import set_trace


def get_tiny_images(image_paths):
  
  feats = []
  number_of_images = len(image_paths)
  #lopping over all images
  for original_img in range(0, number_of_images):
    # Creating grayscale images from their original ones
    img_grayscale = load_image_gray(image_paths[original_img])
    # Resizing grayscale images into 16x16 array (256 pixels)
    img_resized = cv2.resize(img_grayscale, (16,16))
    # Creating normilized image vector
    # (image_pixels - mean_image_pixels)/max(image_pixels)
    img_mean = img_resized.flatten() - np.mean(img_resized.flatten())
    # adding normalied image to the list
    feats.append(img_mean/np.max(np.abs(img_mean)))
  # Returning tiny images
  return np.array(feats)

def build_vocabulary(image_paths, vocab_size):

  print('received vocab size = ' + str(vocab_size))
  dim = 128
  vocab = np.zeros((vocab_size,dim))
  row_factor = 20
  number_of_images = len(image_paths)
  num_of_sift_row = row_factor * number_of_images
  key_points_all = np.zeros((num_of_sift_row, dim))
  step_number = 0
  i = 0
  # looping over all images
  for target_img in image_paths:
    # Grayscale image. It needs to be float not integers
    img_grayscale = load_image_gray(target_img).astype('float32')
    
    # Calculating sift keypints
    vlsd = vlfeat.sift.dsift(img_grayscale,fast=True,step=20)

    # Keypints vector shape
    s1, s2 = vlsd[1].shape

    # Randomly permute a sequence, or return a permuted range.
    permutated_kp = np.random.permutation(s1)

    # looping over all permutated_kp
    for num_step in range(0, row_factor): 
      key_points_all[num_step + step_number, :] = vlsd[1][permutated_kp[num_step], :]

    step_number = step_number + row_factor
    i = i + 1 
  return vlfeat.kmeans.kmeans(key_points_all.astype('float32'), vocab_size)

def get_bags_of_sifts(image_paths, vocab_filename):
  # How to read pkl files in python?
  # Our pkl file is, in fact, a serialized pickle file, which means it has been dumped using Python's pickle module.
  with open(vocab_filename, 'rb') as packed_vocabulary:
    # vocab = pickle.load(f)
    unpacked_vocabulary = pickle.load(packed_vocabulary)
  # vocabulary vector size
  vocab_size_1, vocab_size_2 = unpacked_vocabulary.shape
  print('Vocab Shape')
  print(unpacked_vocabulary.shape)

  # number of images
  number_of_images = len(image_paths)

  # kp to return
  feats = []

  # looping over all images to calculate image histogram
  for target_img in image_paths:

    # grayscale image
    img_grayscale = load_image_gray(target_img).astype('float32')

    # Keyoiunts using sift
    vlsd_place, vlsd_kp  = vlfeat.sift.dsift(img_grayscale,fast=True,step=10)

    # It's nuber should be float type
    vlsd_kp = vlsd_kp.astype('float32')

    # Calculating image histogram
    img_histogram = np.zeros(vocab_size_1)

    # calculating kp distance
    img_pair_dist_all = sklearn_pairwise.pairwise_distances(vlsd_kp, unpacked_vocabulary)
    for img in img_pair_dist_all:
      img_histogram[np.argmin(a=img,axis=0)] = img_histogram[np.argmin(a=img,axis=0)] + 1 
    
    # Normal Error
    norm_error = np.linalg.norm(img_histogram)
    feats.append(img_histogram / norm_error)
  return np.array(feats)

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats, metric='euclidean'):
  test_labels = []

  # Calculating distance between to images using pixelwise
  img_pair_dist_all = sklearn_pairwise.pairwise_distances(train_image_feats, test_image_feats)

  # N = train_image_feats.shape[0]
  a, b = train_image_feats.shape

  # M = test_image_feats.shape[0]
  c, d = test_image_feats.shape

  # looping over training and testing image to find the 1NN to each testing data and labling it to that one 
  for img_test_num in range(0, c):
    # calculating minimum distance od the test image to the train image
    img_min_dist = np.min(img_pair_dist_all[img_test_num,:])
    # Looping over training images to find the nearest one 
    for img_train_num in range(0, a):
      if (img_pair_dist_all[img_test_num,img_train_num] == img_min_dist):
        # Labeling the testing image
        test_labels.append(train_labels[img_train_num])
        break
  return  test_labels

def svm_classify(train_image_feats, train_labels, test_image_feats):

  ground_truth_label = np.zeros(len(train_labels))
  t1, t2 = test_image_feats.shape
  label_result_test = []

  support_vector_machine_s = {}
  for target_class in list(set(train_labels)):
    support_vector_machine_s[target_class] = LinearSVC(random_state=0, tol=1e-3, loss='hinge', C=5)
  
  number_of_classes = len(list(set(train_labels)))
  number_of_train_labels = len(train_labels)

  ceoficients_list = []
  intercepts_list = []

  for i in range(0, number_of_classes):

    for j in range(number_of_train_labels):

      ground_truth_label[j] = (list(set(train_labels))[i]==train_labels[j])

    true_false_labels = np.ones(len(ground_truth_label))*-1
    number_of_true_false_labelss = len(true_false_labels)

    for label_number in range(0, number_of_true_false_labelss):
      if(ground_truth_label[label_number] == 1):
        true_false_labels[label_number] = 1

    support_vector_machine_s[list(set(train_labels))[i]].fit(train_image_feats,true_false_labels,sample_weight=None)

    ceoficients_list.append(support_vector_machine_s[list(set(train_labels))[i]].coef_)
    intercepts_list.append(support_vector_machine_s[list(set(train_labels))[i]].intercept_)
  
  ceoficients_list = np.array(ceoficients_list)
  intercepts_list = np.array(intercepts_list)

  for i in range(0, t1):


    conf_rel_list = []


    for j in range(len(list(set(train_labels)))):
      conf_rel_list.append(np.dot(ceoficients_list[j,0,:], test_image_feats[i, :]) + intercepts_list[j, :])
    

    for label_num in range(0, len(list(set(train_labels)))):

      if(conf_rel_list[label_num] == max(conf_rel_list)):

        label_result_test.append(list(set(train_labels))[label_num])

  return label_result_test