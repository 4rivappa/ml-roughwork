import math
import random
import statistics
from collections import defaultdict

def load_data(file_name):
    file = open(file_name, "r")
    lines = file.readlines()
    return_arr = []
    for line in lines:
        elements = line.split(",")
        if len(elements) == 5:
            for elem_i in range(len(elements)):
                if elem_i == 4:
                    elements[elem_i] = elements[elem_i].replace("\n", "")
                    break
                elements[elem_i] = float(elements[elem_i])
            return_arr.append(elements)
    return return_arr

def load_data_2(filename):
    return_arr = []
    file = open(filename, 'r')
    lines = file.readlines()
    for line in lines:
        elements = line.split(" ")
        for elem_i in range(len(elements)):
            elements[elem_i] = int(elements[elem_i])
        return_arr.append(elements)
    return return_arr

def discretize_data(arr):
    for k in range(len(arr)):
        for kk in range(len(arr[k])):
            if type(arr[k][kk]) == float:
                arr[k][kk] = round(arr[k][kk])

data = load_data("iris.data")
# discretize_data(data)

def divide_into_train_test(data_arr, train_len, test_len):
    random.shuffle(data_arr)
    return data_arr[:train_len], data_arr[train_len:]

# train_data, test_data = divide_into_train_test(data, 120, 30)

# print(test_data)
# print(data)


def get_classes_of_array(arrr):
    class_dict = {}
    for row in arrr:
        if row[-1] in class_dict:
            class_dict[row[-1]] += 1
        else:
            class_dict[row[-1]] = 1
    return_arr = []
    for key, value in class_dict.items():
        return_arr.append([key, value/len(arrr)])
    return return_arr

def populate_big_dict(arrr, big_dict, class_freq_dict):
    for row in arrr:
        for row_i in range(len(row)-1):
            if row[row_i] in big_dict[row_i]:
                if row[-1] in big_dict[row_i][row[row_i]]:
                    big_dict[row_i][row[row_i]][row[-1]] += 1
                    if row[-1] in class_freq_dict[row_i]:
                        class_freq_dict[row_i][row[-1]] += 1
                    else:
                        class_freq_dict[row_i][row[-1]] = 1
                else:
                    big_dict[row_i][row[row_i]][row[-1]] = 1
                    if row[-1] in class_freq_dict[row_i]:
                        class_freq_dict[row_i][row[-1]] += 1
                    else:
                        class_freq_dict[row_i][row[-1]] = 1
            else:
                big_dict[row_i][row[row_i]] = {row[-1]: 1}
                if row[-1] in class_freq_dict[row_i]:
                    class_freq_dict[row_i][row[-1]] += 1
                else:
                    class_freq_dict[row_i][row[-1]] = 1
    # print(class_freq_dict)
    # print(big_dict)
    for col_id_key in big_dict:
        for spec_col_val_key in big_dict[col_id_key]:
            for class_id_key in big_dict[col_id_key][spec_col_val_key]:
                big_dict[col_id_key][spec_col_val_key][class_id_key] /= class_freq_dict[col_id_key][class_id_key]
    # print(big_dict)


def bayes_basic_setup(arr):
    classes = get_classes_of_array(arr)
    # print(classes)
    big_dict = {}
    class_freq_dict = {}
    for k in range(len(arr[0])-1):
        # big_dict[k] = defaultdict([0]*len(classes))
        big_dict[k] = {}
        class_freq_dict[k] = {}
    populate_big_dict(arr, big_dict, class_freq_dict)
    # print(big_dict)
    return classes, big_dict



def calculate_class_bayes(classes, big_dict, test_elem):
    prob_class = {}
    for k in range(len(test_elem)):
        if k in big_dict:
            if test_elem[k] in big_dict[k]:
                for class_key in big_dict[k][test_elem[k]]:
                    if class_key in prob_class:
                        prob_class[class_key] *= big_dict[k][test_elem[k]][class_key]
                    else:
                        prob_class[class_key] = big_dict[k][test_elem[k]][class_key]
    for cl in classes:
        if cl[0] in prob_class:
            prob_class[cl[0]] *= cl[1]
    max_key = max(prob_class, key=prob_class.get)
    return max_key

        # for ll in big_dict:
            # for kk in big_dict[ll]:
                # for jj in big_dict[ll][kk]:

def get_accuracy(classes, big_dict, test_data):
    correct_count = 0
    for element in test_data:
        predicted = calculate_class_bayes(classes, big_dict, element)
        if predicted == element[-1]:
            correct_count += 1
    if len(test_data) == 0:
        return 0
    return correct_count/len(test_data)



train_data, test_data = divide_into_train_test(data, 120, 30)
    

classes_arr, big_dictionary = bayes_basic_setup(train_data)

print("Without discritization - Iris divided into 120, 30: ")

print("Accuracy of test data with train data is: " + str(get_accuracy(classes_arr, big_dictionary, test_data)*100) + "%")



discretize_data(data)

train_data, test_data = divide_into_train_test(data, 120, 30)


classes_arr, big_dictionary = bayes_basic_setup(train_data)

print("With discritization - Iris divided into 120, 30: ")

print("Accuracy of test data with train data is: " + str(get_accuracy(classes_arr, big_dictionary, test_data)*100) + "%")



train_data = load_data_2('./pp_tra.dat')
test_data = load_data_2('./pp_tes.dat')


classes_arr, big_dictionary = bayes_basic_setup(train_data)

print("COE Data: ")

print("Accuracy of test data with train data is: " + str(get_accuracy(classes_arr, big_dictionary, test_data)*100) + "%")

