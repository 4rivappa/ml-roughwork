import statistics
import random

def load_data(filename):
    return_arr = []
    file = open(filename, 'r')
    lines = file.readlines()
    for line in lines:
        elements = line.split(" ")
        for elem_i in range(len(elements)):
            elements[elem_i] = int(elements[elem_i])
        return_arr.append(elements)
    return return_arr

def minkowski_cal(point1, point2, p):
    length = len(point1)
    total_sum = 0
    for i in range(length-1):
        total_sum += pow(abs(point1[i] - point2[i]), p)
    result = pow(total_sum, 1/p)
    return result

def cal_euclid_dist(point1, point2):
    length = len(point1)
    total_sum = 0
    for i in range(length-1):
        total_sum += pow(point1[i] - point2[i], 2)
    result = pow(total_sum, 1/2)
    return result

def cal_dist_array(data_arr, point):
    return_arr = []
    for index in range(len(data_arr)):
        dist = cal_euclid_dist(data_arr[index], point)
        return_arr.append([dist, index])
    # sorted(return_arr, key = lambda x : x[0])
    return_arr.sort(key = lambda x: x[0])
    return return_arr

def cal_k_nnc(data_arr, k_val, point):
    short_dists = []
    sorted_dists = cal_dist_array(data_arr, point)
    for i in range(k_val):
        short_dists.append(data_arr[sorted_dists[i][1]][-1])
    class_count = []
    for i in range(9):
        class_count.append([short_dists.count(i), i])
    class_count.sort(key = lambda x: x[0])
    return class_count[-1][1]



train_data = load_data('./pp_tra.dat')
test_data = load_data('./pp_tes.dat')


def divide_into_three(data_arr):
    arrays = []
    length = len(data_arr)
    random.shuffle(data_arr)
    prev = 0
    for i in range(3):
        arrays.append(data_arr[prev:prev + length//3])
        prev = prev + length//3
    return arrays

three_folds = divide_into_three(train_data)


def get_accuracy_train_test(train_data, test_data, knn_k_val):
    total_test = len(test_data)
    correct_test = 0
    for k in test_data:
        ret_str = cal_k_nnc(train_data, knn_k_val, k)
        if ret_str == k[-1]:
            correct_test += 1
    accuracy = (correct_test/total_test)*100
    return accuracy


def k_fold_method(folds):
    k = len(folds)
    averages = []
    std_devs = []
    for tt in range(1, 20):
        accuracy_arr = []
        std_dev_arr = []
        for i in range(k):
            arr = []
            for j in range(k):
                if i != j:
                    arr.extend(folds[j])
            accuracy = get_accuracy_train_test(folds[i], arr, tt)
            accuracy_arr.append(accuracy)
            std_dev_arr.append(accuracy)
        accuracy_avg = sum(accuracy_arr)/len(accuracy_arr)
        std_dev = statistics.stdev(std_dev_arr)
        std_devs.append(std_dev)
        averages.append(accuracy_avg)
    return averages, std_devs


avgs, std_devs = k_fold_method(three_folds)
best_k_val_for_knn = avgs.index(max(avgs)) + 1



def mink_cal_dist_array(data_arr, point, p):
    return_arr = []
    for index in range(len(data_arr)):
        dist = minkowski_cal(data_arr[index], point, p)
        return_arr.append([dist, index])
    # sorted(return_arr, key = lambda x : x[0])
    return_arr.sort(key = lambda x: x[0])
    return return_arr

def mink_cal_k_nnc(data_arr, k_val, point, p):
    short_dists = []
    sorted_dists = mink_cal_dist_array(data_arr, point, p)
    for i in range(k_val):
        short_dists.append(data_arr[sorted_dists[i][1]][-1])
    class_count = []
    for i in range(9):
        class_count.append([short_dists.count(i), i])
    class_count.sort(key = lambda x: x[0])
    return class_count[-1][1]

def mink_get_accuracy_train_test(train_data, test_data, knn_k_val, p):
    total_test = len(test_data)
    correct_test = 0
    for k in test_data:
        ret_str = mink_cal_k_nnc(train_data, knn_k_val, k, p)
        if ret_str == k[-1]:
            correct_test += 1
    accuracy = (correct_test/total_test)*100
    return accuracy

def find_best_p_mink(train_data, test_data, best_k_in_knn):
    p_accuracy = []
    for p in range(4):
        p_accuracy.append(mink_get_accuracy_train_test(train_data, test_data, best_k_in_knn, p+1))
    return p_accuracy.index(max(p_accuracy))+1, max(p_accuracy)


best_p_val_for_mink, max_accuracy_for_p = find_best_p_mink(train_data, test_data, best_k_val_for_knn)


print("Best k value for knn in 3 fold cross validation for ocr dataset is: " + str(best_k_val_for_knn))
print("Best P value for minkowski metric for test-set using best knn classifier is: " + str(best_p_val_for_mink))
print("Accuracy for minkowski algorithm, with P value " + str(best_p_val_for_mink) + " is: " + str(max_accuracy_for_p))
