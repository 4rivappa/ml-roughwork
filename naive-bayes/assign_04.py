from decimal import ROUND_HALF_DOWN
import math
import random
import statistics
import numpy as np
from collections import defaultdict

categorical = [1,2,5,6,8,10,11,12]

def load_data(file_name):
    file = open(file_name, "r")
    lines = file.readlines()
    lines = lines[1:]
    return_arr = []
    for line in lines:
        elements = line.split(",")
        if len(elements) == 14:
            for elem_i in range(len(elements)):
                elements[elem_i] = float(elements[elem_i])
            return_arr.append(elements)
    return return_arr

def divide_into_train_test(data_arr, train_percent):
    random.shuffle(data_arr)
    end_point = round(len(data_arr) * (train_percent/100))
    return data_arr[:end_point], data_arr[end_point:]

def get_classes_of_array(arrr):
    class_dict = {}
    for row in arrr:
        if row[-1] in class_dict:
            class_dict[row[-1]] += 1
        else:
            class_dict[row[-1]] = 1
    return_dict = {}
    for key, value in class_dict.items():
        return_dict[key] = value/len(arrr)
    return return_dict


def populate_big_dict(arrr, big_dict, class_freq_dict):
    cont_cat_dict = {}
    for row in arrr:
        for row_i in range(len(row)-1):
            if row_i in categorical and row[row_i] in big_dict[row_i]:
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
            elif row_i in categorical:
                big_dict[row_i][row[row_i]] = {row[-1]: 1}
                if row[-1] in class_freq_dict[row_i]:
                    class_freq_dict[row_i][row[-1]] += 1
                else:
                    class_freq_dict[row_i][row[-1]] = 1
            if row_i not in categorical:
                if row_i in cont_cat_dict:
                    if row[-1] in cont_cat_dict[row_i]:
                        cont_cat_dict[row_i][row[-1]].append(row[row_i])
                    else:
                        cont_cat_dict[row_i][row[-1]] = [row[row_i]]
                else:
                    cont_cat_dict[row_i] = {}
                    cont_cat_dict[row_i][row[-1]] = [row[row_i]]
    
    # print(cont_cat_dict)
    for col_key in cont_cat_dict:
        for class_key in cont_cat_dict[col_key]:
            curr_list = cont_cat_dict[col_key][class_key]
            list_std = np.std(curr_list)
            list_mean = np.mean(curr_list)
            cont_cat_dict[col_key][class_key] = [pow(list_std, 2), list_mean]
    # print(cont_cat_dict)

    for col_id_key in big_dict:
        for spec_col_val_key in big_dict[col_id_key]:
            for class_id_key in big_dict[col_id_key][spec_col_val_key]:
                big_dict[col_id_key][spec_col_val_key][class_id_key] /= class_freq_dict[col_id_key][class_id_key]

    # return cont_dict
    return cont_cat_dict


def bayes_basic_setup(arr):
    big_dict = {}
    class_freq_dict = {}
    for k in range(len(arr[0])-1):
        big_dict[k] = {}
        class_freq_dict[k] = {}
    # print(big_dict)
    add_cont_dict = populate_big_dict(arr, big_dict, class_freq_dict)
    # print(big_dict)
    return big_dict, add_cont_dict


def calculate_class_bayes(class_prob, big_dict, cont_dict, test_elem):
    prob_class = {}
    for k in range(len(test_elem)):
        if k in big_dict:
            if k in categorical:
                if test_elem[k] in big_dict[k]:
                    for class_key in big_dict[k][test_elem[k]]:
                        if class_key in prob_class:
                            prob_class[class_key] *= big_dict[k][test_elem[k]][class_key]
                        else:
                            prob_class[class_key] = big_dict[k][test_elem[k]][class_key]
            else:
                for class_key in cont_dict[k]:
                    if class_key in prob_class:
                        prob_class[class_key] *= (1/(2*math.sqrt(2*3.14*cont_dict[k][class_key][0])))*pow(2.718, (pow((test_elem[k] - cont_dict[k][class_key][1]), 2) / (2*cont_dict[k][class_key][0])))
                    else:
                        prob_class[class_key] = (1/(2*math.sqrt(2*3.14*cont_dict[k][class_key][0])))*pow(2.718, (pow((test_elem[k] - cont_dict[k][class_key][1]), 2) / (2*cont_dict[k][class_key][0])))

    for class_key in prob_class:
        if class_key in class_prob:
            prob_class[class_key] *= class_prob[class_key]
    max_key = max(prob_class, key=prob_class.get)
    return max_key


def get_accuracy(class_prob, big_dict, cont_dict, test_data):
    correct_count = 0
    for element in test_data:
        predicted = calculate_class_bayes(class_prob, big_dict, cont_dict, element)
        if predicted == element[-1]:
            correct_count += 1
    if len(test_data) == 0:
        return 0
    return (correct_count/len(test_data)) * 100





data = load_data("heart.csv")
# print(data[0:10])

train_set, test_set = divide_into_train_test(data, 70)
# print(len(train_set), len(test_set))

class_prob = get_classes_of_array(train_set)
# print(class_prob)

big_dictionary, cont_dictionary = bayes_basic_setup(train_set)
# print(big_dictionary)
# print(cont_dictionary)

# print(get_accuracy(class_prob, big_dictionary, cont_dictionary, test_set))

print("Accuracy of test data with train data for naive bayes theorem is : " + str(get_accuracy(class_prob, big_dictionary, cont_dictionary, test_set)))
