import os, time, sys
from PIL import Image
from PIL import Image as im

# Linear Algebra Library
from numpy import *
from numpy.linalg import *
import numpy as np
import numpy.linalg as la

import math
#from cvxopt import *
import cvxopt
import cvxopt.solvers

# Take Training images now
path = os.path.abspath(__file__)
path_training = path.rsplit('\\', 1)[0] + '\\'  + "att_faces_train"
path_testing  = path.rsplit('\\', 1)[0] + '\\'  + "att_faces_test"


class kernels:
    def __init__(self, kernel = None):
        self.name =  kernel

        def Gauss(a, mu, std_dev):
            return math.exp(-la.norm(a-mu)/ (2 * math.pow(std_dev,2)))

        def polynomial(a, b, degree):
            return math.pow((np.dot(a, b)), degree)

def Gauss(a, mu, std_dev):
            return math.exp(-la.norm(a-mu)/ (2 * math.pow(std_dev,2)))

def linear(A, B):
    return np.dot(A,B)

class Wide_Streets_among_classes:
    def __init__(self, name, vector):
        self.name = name
        self.vector = vector

def quadratic_solver(P, q, G, h, A, b):
    opt = cvxopt.solvers.qp(P, q, G, h, A, b)
    result = np.ravel(opt['x'])
    return result

def get_support_vectors(Train_classifier, labler, result):
    #print result
    #works for linear kernel
    #support  = result > 1e-10
    support  = result > 1e3

    #support  = result > 1e-1
    more_than_one = np.arange(len(result))[support]
    support_first = result[support]
    support_first_vect = Train_classifier[support]
    labler = labler.transpose()
    #print support

    support_first_y = labler[support]

    #sys.exit()
    print 'The number of the Support Vectors for this data : ' + str(len(support_first_vect))
    return support_first, support_first_vect, support_first_y, support, more_than_one

def get_b(support_first, support_first_vect, support_first_y,support, more_than_one, N):
    b = 0
    for each in range(len(support_first)):
        b += support_first_y[each]
        now = support_first_y[each]
        x = 0
        for each1 in range(len(support_first_vect)):
            #x += (support_first[each1] * support_first_y[each1] * N[more_than_one[each], support])

            #x += (support_first[each1] * support_first_y[each1] * Gauss(support_first_vect[each], support_first_vect[each1], 100000))
            x += (support_first[each1] * support_first_y[each1] * linear(support_first_vect[each], support_first_vect[each1]))

            #Gauss(single_image, vect, 10000)
        #j = np.sum(support_first * support_first_y * N[more_than_one[each], support])
        #j = np.sum(support_first * support_first_y * Gauss(support_first_vect[each],support_first_vect, 10000))
        b = b - x
    b /= len(support_first)
    return b

def get_gram_matix(Train_classifier):
    number, just = Train_classifier.shape
    N = np.zeros((number, number))
    for each in range(number):
        for e in range(number):
            #N[each][e] = Gauss(Train_classifier[each], Train_classifier[e], 100000)
            N[each][e] = linear(Train_classifier[each], Train_classifier[e])
    return N, number

def get_train_results(Train_classifier, labler):
    N, number = get_gram_matix(Train_classifier)

    X = np.outer(labler, labler)

    X1 = X * N
    P = cvxopt.matrix(X1)

    X = np.ones(number)
    X2 = X * (-1)
    q = cvxopt.matrix(X2)

    A = cvxopt.matrix(labler, (1,number))
    b = cvxopt.matrix(0.0)

    G = cvxopt.matrix(np.diag(np.ones(number) * -1))
    h = cvxopt.matrix(np.zeros(number))

    solution = quadratic_solver(P, q, G, h, A, b)
    support_first, support_first_vect, support_first_y, support, more_than_one = get_support_vectors(Train_classifier, labler, solution)
    b = get_b(support_first, support_first_vect, support_first_y,support, more_than_one, N)

    return solution, support_first, support_first_vect, support_first_y, support, more_than_one, b

def to_new_space(single_image, support_first, support_first_y ,support_first_vect, support, b):
                 #result, support, y, more_than_one, b):
    expectation = np.zeros(len(single_image))
    add = 0
    for alpha, yi, vect in zip(support_first, support_first_y, support_first_vect):
            #add += alpha * yi * Gauss(single_image, vect, 100000)
            add += alpha * yi * linear(single_image, vect)
    expectation = add
    '''
    for each in range(len(single_image)):
        add = 0
        for alpha, yi, vect in zip(support_first, support_first_y, support_first_vect):
            add += alpha * yi * Gauss(single_image[each], vect, 10000)
        expectation[each] = add
    '''
    return (expectation + b)

def classify_new(single_image, support_first, support_first_y ,support_first_vect, support, b):
    return to_new_space(single_image, support_first, support_first_y ,support_first_vect, support, b)

def get_labeler(each, length):
    l = np.ones((1,length))
    l = (l * -1)
    l[0, (each * 5)] = 1
    l[0, ((each * 5) + 1)] = 1
    l[0, ((each * 5) + 2)] = 1
    l[0, ((each * 5) + 3)] = 1
    l[0, ((each * 5) + 4)] = 1
    return l

class Eigen_Face:
    def __init__(self, name):
        self.name           = name
        self.A              = np.reshape(([0] * 10304), [-1,1])
        self.eig_val        = None
        self.eig_vect       = None
        self.new_eig_val    = None
        self.new_eig_vect   = None
        self.mult_val       = None
        self.mean_A         = [0] * 10304
        self.AT             = None


    def set_name(self, _name):
        self.name = _name

    def get_name(self):
        return (self._name)

    def mod_A(self, add):
        self.A = np.append(self.A, add, axis=1)

    def correct_A(self):
        self.A = self.A[0:,1:]
        #print self.A.shape

    def find_mean_face_A(self):
        for each in range(len(self.A)):
            self.mean_A[each] = float(float(self.A[each,:].sum())/len(self.A[0,:]))

    def subtract_mean_face_A(self):
        for each in range(len(self.A)):
            self.A[each,:] -= float(float(self.A[each, :].sum())/len(self.A[0,:]))

    def transpose_A(self):
        self.AT = np.transpose(self.A)

    def AT_prod_A(self):
        self.prod = np.dot(self.AT,self.A)

    def set_eig_val_vect(self):
        self.eig_val , self.eig_vect = np.linalg.eig(self.prod)

    def sort_eigen_vals_decreasingly(self):
        self.index = np.argsort(-self.eig_val)
        #print self.index

    def set_sorted_eigen_values_vectors(self):
        self.new_eig_val  = self.eig_val[self.index]
        self.new_eig_vect = self.eig_vect[:,self.index]
        #print self.new_eig_val

    def multipy_new_eigen_vectors_by_A(self):
        self.mult_val = np.dot(self.A, self.new_eig_vect)

    def normalize_the_new_eigen_vectors(self):
        #print self.mult_val.shape
        #print len(self.mult_val[0])
        #print self.mult_val
        for each in range (len(self.mult_val[0])):
            self.mult_val[:,each] /= la.norm(self.mult_val[:,each])

        #print self.mult_val.shape

    def get_mean_face_A(self):
        return self.mean_A

    def transpose_normalized(self):
        self.Pprime = np.transpose(self.mult_val)
        #print self.Pprime.shape

    def select_most_important_P(self):
        self.P = self.Pprime[:,:50]
        #print self.P.shape

    def get_A(self):
        return self.A

    def get_AT(self):
        return self.AT

    def get_prod(self):
        return self.prod

    def get_eig_val(self):
        return self.eig_val

    def get_eig_vect(self):
        return self.eig_vect

    def get_eig_val_vect(self):
        return self.eig_val , self.eig_vect

    def get_sorted_index(self):
        return self.index

    def get_sorted_eigen_values(self):
        return self.new_eig_val

    def get_sorted_eigen_vectors(self):
        return self.new_eig_vect

    def get_sorted_eigen_values_vectors(self):
        return self.new_eig_val, self.new_eig_vect

    def get_mult_A(self):
        return self.mutlt_val

    def get_transpose_normalized(self):
        return self.Pprime

    def get_P(self):
        return self.P

    def get_Pprime(self):
        return self.Pprime



def Train(path):
    svm_train_face = Eigen_Face('svm')
    names = []
    '''Reading each image and convering them to an array'''
    for each in os.listdir(path):
        for each1 in os.listdir(path + '\\' + each):
            names.append(each)
            im = Image.open(path + '\\' + each + '\\' + each1)
            get_image = im.getdata()
            column_image = np.reshape(get_image, [-1,1])
            #print column_image.shape
            svm_train_face.mod_A(column_image)

    svm_train_face.correct_A()
    svm_train_face.find_mean_face_A()
    svm_train_face.subtract_mean_face_A()
    svm_train_face.transpose_A()

    svm_train_face.AT_prod_A()
    svm_train_face.set_eig_val_vect()
    svm_train_face.sort_eigen_vals_decreasingly()
    svm_train_face.set_sorted_eigen_values_vectors()
    svm_train_face.multipy_new_eigen_vectors_by_A()
    svm_train_face.normalize_the_new_eigen_vectors()

    svm_train_face.transpose_normalized()

    svm_train_face.select_most_important_P()
    y = svm_train_face.get_P()
    print y.shape
    
    #print svm_train_face.get_AT().shape
    _vectors_machines = {}
    for each in range(40):
        labler = get_labeler(each, len(names))
        store = {}
        solution, support_first, support_first_vect, support_first_y, support, more_than_one, b = get_train_results(y,labler)
        store['solution'] = solution
        store['support_first'] = support_first
        store['support_first_vect'] = support_first_vect
        store['support_first_y'] = support_first_y
        store['support'] = support
        store['more_than_one'] = more_than_one
        store['b'] = b
        _vectors_machines[names[each*5]] = store
        print 'Trained Support Vector Machine for Class: ' + str(names[each*5])
        print 'So far had trained ' + str(each + 1) + 'SVM\'s'
        print 'Please be patient. ' + str(40-each-1) + ' to go...............'
        #break
    #print _vectors_machines
    return _vectors_machines

def Testing(path, _vectors_machines):
    svm_test_face = Eigen_Face('test_svm')
    names = []
    '''Reading each image and convering them to an array'''
    for each in os.listdir(path):
        for each1 in os.listdir(path + '\\' + each):
            names.append(each)
            im = Image.open(path + '\\' + each + '\\' + each1)
            get_image = im.getdata()
            column_image = np.reshape(get_image, [-1,1])            
            svm_test_face.mod_A(column_image)

    svm_test_face.correct_A()
    
    svm_test_face.find_mean_face_A()
    svm_test_face.subtract_mean_face_A()
    
    svm_test_face.transpose_A()

    svm_test_face.AT_prod_A()
    svm_test_face.set_eig_val_vect()
    svm_test_face.sort_eigen_vals_decreasingly()
    svm_test_face.set_sorted_eigen_values_vectors()
    svm_test_face.multipy_new_eigen_vectors_by_A()
    svm_test_face.normalize_the_new_eigen_vectors()

    svm_test_face.transpose_normalized()
    svm_test_face.select_most_important_P()
    
    test_data = svm_test_face.get_P()
    print test_data.shape
    l_a = []
    for key in _vectors_machines.keys():
        l_a.append(key)
    i = 0
    yes = 0
    no =  0
    for each in test_data:
        #print each.shape
        #sys.exit()
        values = []
        maximum = -999999999999999999999999999999
        for key in _vectors_machines:
            val = classify_new(each, _vectors_machines[key]['support_first'], _vectors_machines[key]['support_first_y'], _vectors_machines[key]['support_first_vect'], _vectors_machines[key]['support'], _vectors_machines[key]['b'])
            #alues.append(val)

            #print val
            if(val > maximum ):
                maximum = val
                k = key

        #print k
        print 'origianl: ' + str(names[i]) + ' Classified: ' + str(k)
        if(names[i] == k):
            yes += 1
        else:
            no += 1
        i += 1
    print 'Yes is: ' + str(yes)
    print 'No is : ' + str(no)
    print 'percentage: ' + str(float(yes)/2)
        #print values
        #break


start_time = time.time()
machines = Train(path_training)
Testing(path_testing, machines)
end_time = time.time()
print 'time taken = ' + str((end_time - start_time)/60) + ' minutes'
