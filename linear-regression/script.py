import statistics
import random

def load_data(filename):
    return_arr = []
    file = open(filename, 'r')
    lines = file.readlines()
    category_line = False
    for line in lines:
        elements = line.split(",")
        if category_line == False:
            return_arr.append(elements)
            category_line = True
            continue
        for elem_i in range(len(elements)):
            elements[elem_i] = float(elements[elem_i])
        return_arr.append(elements)
    return return_arr[0], return_arr[1:]

def divide_train_test(data_arr, train_percentage):
    random.shuffle(data_arr)
    divide_point = (train_percentage * len(data_arr))//100
    return data_arr[:divide_point], data_arr[divide_point:]


def cal_expected_values(data_arr, coeff_arr):
    return_values_arr = []
    for row in data_arr:
        curr_expected_value = 0
        for i in range(len(row)):
            if i == len(row)-1:
                curr_expected_value += coeff_arr[i]
                continue
            curr_expected_value += row[i]*coeff_arr[i]
        return_values_arr.append(curr_expected_value)
    return return_values_arr

def cal_tss(data_arr):
    sum_arr = [0]*len(data_arr[0])
    for ii in range(len(data_arr)):
        for jj in range(len(data_arr[ii])):
            sum_arr[jj] += data_arr[ii][jj]
    mean_arr = []
    for sum in sum_arr:
        mean_arr.append(sum/len(data_arr))
    tss_sum = 0
    for ii in range(len(data_arr)):
        tss_sum += pow((data_arr[ii][-1] - mean_arr[-1]), 2)
    return tss_sum, mean_arr

def cal_sse_mse_r2(data_arr, new_predicted_arr):
    return_sse = 0
    for ii in range(len(data_arr)):
        diff = data_arr[ii][-1] - new_predicted_arr[ii]
        return_sse += diff * diff
    tss, mean_arr = cal_tss(data_arr)
    r_two = (1 - (return_sse/tss))
    return return_sse, return_sse/len(new_predicted_arr), r_two

def cal_difference_arr(data_arr, new_predicted_arr):
    return_diff_arr = []
    for ii in range(len(data_arr)):
        return_diff_arr.append(new_predicted_arr[ii] - data_arr[ii][-1])
    return return_diff_arr

# for formula refer https://www.geeksforgeeks.org/gradient-descent-in-linear-regression/

def update_my_coeff_arr(data_arr, diff_arr, coeff_arr, alpha_val):
    sigma_arr = [0]*len(coeff_arr)
    for ii in range(len(data_arr)):
        for jj in range(len(data_arr[ii])):
            if jj == len(data_arr[ii])-1:
                sigma_arr[jj] += diff_arr[ii]
                continue
            sigma_arr[jj] += diff_arr[ii] * data_arr[ii][jj]
    subtract_arr = []
    for sum in sigma_arr:
        subtract_arr.append(alpha_val * (sum / len(data_arr)))
        # subtract_arr.append(alpha_val * (sum))
    for kk in range(len(subtract_arr)):
        coeff_arr[kk] -= subtract_arr[kk]
    return coeff_arr


def main_linear_regression(data_arr, iterator_count, alpha_val):
    # randomly assigning coefficient values
    coeff_arr = []
    tss, mean_of_data = cal_tss(data_arr)
    for k in range(len(data_arr[0])):
        coeff_arr.append(mean_of_data[-1]/(mean_of_data[k] * 9))
        # coeff_arr.append((mean_of_data[-1]/mean_of_data[k])*(random.uniform(0, 1)))
        # coeff_arr.append(random.uniform(0, 1))
    # iterations of model updation
    curr_iter_count = 0
    while curr_iter_count < iterator_count:
        predicted_val_arr = cal_expected_values(data_arr, coeff_arr)
        diff_arr = cal_difference_arr(data_arr, predicted_val_arr)
        coeff_arr = update_my_coeff_arr(data_arr, diff_arr, coeff_arr, alpha_val)
        curr_iter_count += 1
    # printing coefficients after training
    print("Coefficients of trained model are: ")
    print(coeff_arr)
    new_predicted_values = cal_expected_values(data_arr, coeff_arr)
    std_sse, std_mse, std_r_two = cal_sse_mse_r2(data_arr, new_predicted_values)
    print("==============================================================================")
    print("The SSE for the training data set using model coefficients is: ")
    print(std_sse)
    print("The MSE for the training data set using model coefficients is: ")
    print(std_mse)
    print("The r2 coefficient for training data set using model coefficients is: ")
    print(std_r_two)
    print("==============================================================================")
    return coeff_arr

def testing_for_test_data(data_arr, model_coefficients):
    predicted_values = cal_expected_values(data_arr, model_coefficients)
    std_sse, std_mse, std_r_two = cal_sse_mse_r2(data_arr, predicted_values)
    print("==============================================================================")
    print("The SSE for the test data set using model coefficients is: ")
    print(std_sse)
    print("The MSE for the test data set using model coefficients is: ")
    print(std_mse)
    print("The r2 coefficient for test data set using model coefficients is: ")
    print(std_r_two)
    print("==============================================================================")

cat_arr, data_arr = load_data("House Price.csv")
# print(cat_arr, data_arr)

train_data, test_data = divide_train_test(data_arr, 70)


## u can modify iterations count and alpha value here....
#  as there are so many attributes, alpha value plays a major role in choosing the correct point of contact..
model_coeff_arr = main_linear_regression(train_data, 30, 0.000000000000001)

testing_for_test_data(test_data, model_coeff_arr)
