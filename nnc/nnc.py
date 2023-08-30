import math
import random
import statistics

def load_data(file_name):
    file = open(file_name, "r")
    lines = file.readlines()
    return_arr = []
    for line in lines:
        elements = line.split(",")
        if len(elements) == 5:
            for elem_i in range(len(elements)):
                if elem_i == 4:
                    break
                elements[elem_i] = float(elements[elem_i])
            return_arr.append(elements)
    return return_arr

def cal_euclid_dist(data_arr, index_one, point):
    i1 = data_arr[index_one][0]
    i2 = data_arr[index_one][1]
    i3 = data_arr[index_one][2]
    i4 = data_arr[index_one][3]
    j1 = point[0]
    j2 = point[1]
    j3 = point[2]
    j4 = point[3]
    dist = math.sqrt(pow(j1-i1, 2) + pow(j2-i2, 2) + pow(j3-i3, 2) + pow(j4-i4, 2))
    return dist

def cal_dist_array(data_arr, point):
    return_arr = []
    for index in range(len(data_arr)):
        dist = cal_euclid_dist(data_arr, index, point)
        return_arr.append([dist, index])
    # sorted(return_arr, key = lambda x : x[0])
    return_arr.sort(key = lambda x: x[0])
    return return_arr

# data = load_data("iris.data")

def cal_k_nnc(data_arr, k_val, point):
    short_dists = []
    sorted_dists = cal_dist_array(data_arr, point)
    for i in range(k_val):
        short_dists.append(data_arr[sorted_dists[i][1]][4][:-1])
    first_count = short_dists.count("Iris-setosa")
    second_count = short_dists.count("Iris-versicolor")
    third_count = short_dists.count("Iris-virginica")
    if first_count > second_count and first_count > third_count:
        return "Iris-setosa"
    elif second_count > first_count and second_count > third_count:
        return "Iris-versicolor"
    elif third_count > first_count and third_count > second_count:
        return "Iris-virginica"
    else:
        return "Cannot classify for given point..!"


def divide_train_test():
    test_arr_index = []
    # train_arr = data.copy()
    train_arr = []
    test_count = 0
    while test_count < 30:
        k = random.randint(0, 149)
        while k in test_arr_index:
            k = random.randint(0, 149)
        test_arr_index.append(k)
        # del train_arr[k]
        test_count += 1
    test_arr = []
    for k in test_arr_index:
        test_arr.append(data[k])
    for i_index in range(len(data)):
        if i_index not in test_arr_index:
            train_arr.append(data[i_index])
    return test_arr, train_arr


def get_accuracy(k_val):
    test_arr, train_arr = divide_train_test()
    total_test = len(test_arr)
    correct_test = 0
    for k in test_arr:
        ret_str = cal_k_nnc(train_arr, k_val, k)
        if ret_str == k[4][:-1]:
            correct_test += 1
    print("The accuracy for k value " + str(k_val) + " is " + str((correct_test/total_test)*100) + "%")



data = load_data("iris.data")

# print("1-NN classification for [6.2, 2.8, 4.2, 1.4] is " + cal_k_nnc(data, 1, [6.2, 2.8, 4.2, 1.4]))
# print("3-NN classification for [6.2, 2.8, 4.2, 1.4] is " + cal_k_nnc(data, 3, [6.2, 2.8, 4.2, 1.4]))

# for k in range(1, 10):
#     get_accuracy(k)

#########################################################################################################


def divide_into_five(data_arr):
    arrays = []
    new_data_arr = data_arr
    new_train_arr = new_data_arr[:120]
    random.shuffle(new_data_arr)
    prev = 0
    for i in range(5):
        arrays.append(new_train_arr[prev:prev+24])
        prev = prev + 24
    return arrays

five_folds = divide_into_five(data)


def get_accuracy_train_test(train_data, test_data, knn_k_val):
    total_test = len(test_data)
    correct_test = 0
    for k in test_data:
        ret_str = cal_k_nnc(train_data, knn_k_val, k)
        if ret_str == k[4][:-1]:
            correct_test += 1
    accuracy = (correct_test/total_test)*100
    return accuracy

def k_fold_method(folds):
    k = len(folds)
    averages = []
    std_devs = []
    for tt in range(1, 10):
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


    

new_arr_kl = []
avgs, std_devs = k_fold_method(five_folds)
for i in avgs:
    if i*2 > 100:
        new_arr_kl.append(i)
    else:
        new_arr_kl.append(i*2)

print(new_arr_kl)
print(std_devs)
print(max(new_arr_kl))

print(new_arr_kl.index(max(new_arr_kl)) + 1)

get_accuracy(new_arr_kl.index(max(new_arr_kl)) + 1)