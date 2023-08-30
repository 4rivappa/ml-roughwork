import numpy as np
import matplotlib.pyplot as plt


n_vals = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

# test n values
# n_vals = [100, 200, 300]

# all changeable attributes
train_sets_count = 10
test_size = 100
classes_count = 2
knn_k_val_const = 1

mean_1 = [0, 0]
mean_2 = [0, 2]
cov = [[1, 0], [0, 1]]

# concept ---
# cov is covariance matrix
# cov[x][y] represents the covariance between x and y
# covariance of itself, cov[x][x] is variance of x that is square of std(x)


# testing ---
# pts = np.random.multivariate_normal(mean_1, cov, size=100)
# col1 = [[1]]*100
# col1 = np.array(col1)
# print(type(col1))
# print(col1)
# pts2 = np.random.multivariate_normal(mean_2, cov, size=100)



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
    for i in range(classes_count):
        class_count.append([short_dists.count(i), i])
    class_count.sort(key = lambda x: x[0])
    return class_count[-1][1]

def normal_knn_bias(train_data, test_data, knn_k_val):
    # total_test = len(test_data)
    # correct_test = 0
    normal_predicted_class = []
    for k in test_data:
        ret_str = cal_k_nnc(train_data, knn_k_val, k)
        normal_predicted_class.append(ret_str)
        # if ret_str == k[-1]:
        #     correct_test += 1
    # accuracy = (correct_test/total_test)*100
    # return accuracy
    return normal_predicted_class


def normal_knn_bias_2(train_data, test_data, knn_k_val):
    # total_test = len(test_data)
    # correct_test = 0
    normal_predicted_class = []
    for k in test_data:
        ret_str = cal_k_nnc(train_data, knn_k_val, k)
        normal_predicted_class.append([ret_str])
        # if ret_str == k[-1]:
        #     correct_test += 1
    # accuracy = (correct_test/total_test)*100
    # return accuracy
    return normal_predicted_class




def calculate_bias_variance_of_n(n):
    total = n*train_sets_count + test_size
    # print(total)
    class1 = np.random.multivariate_normal(mean_1, cov, size = total//2)
    class2 = np.random.multivariate_normal(mean_2, cov, size = (total - total//2))
    # print(class1, class2)

    class1 = np.append(class1, np.array([[0]]*(total//2)), axis=1)
    class2 = np.append(class2, np.array([[1]]*(total - total//2)), axis=1)
    # print(class1, class2)

    data = np.concatenate((class1, class2), axis=0)
    # print(data)
    np.random.shuffle(data)

    # print(data)

    test_data = data[:test_size]
    train_data = data[test_size:]

    normal_bias_classes = normal_knn_bias(train_data, test_data, knn_k_val_const)
    
    distribute_start = 0
    # total_distribute_classes = np.array([])
    total_distribute_classes = None
    total_none = True
    for i in range(1, train_sets_count+1):
        part_of_train = train_data[distribute_start:n*i]
        return_classes = normal_knn_bias_2(part_of_train, test_data, knn_k_val_const)
        if total_none and total_distribute_classes == None:
            total_distribute_classes = return_classes
            total_none = False
            continue
        if total_none == False:
            total_distribute_classes = np.append(total_distribute_classes, return_classes, axis=1)
    
    # print(normal_bias_classes)
    # print(total_distribute_classes)

    main_prediction = []
    for row in total_distribute_classes:
        class_count = []
        for i in range(classes_count):
            # class_count.append([row.count(i), i])
            class_count.append([np.count_nonzero(row == i), i])
        class_count.sort(key = lambda x: x[0])
        main_prediction.append(class_count[-1][1])
    
    # print(main_prediction)

    bias_sum = 0
    for i in range(len(normal_bias_classes)):
        if normal_bias_classes[i] != main_prediction[i]:
            bias_sum += 1
    bias_avg = bias_sum/len(normal_bias_classes)

    variance_sum = 0
    for i in range(len(total_distribute_classes)):
        row = total_distribute_classes[i]
        curr_count = 0
        for elem in row:
            if elem != main_prediction[i]:
                curr_count += 1
        curr_variance = curr_count/len(row)
        variance_sum += curr_variance
    variance_avg = variance_sum/len(total_distribute_classes)

    return bias_avg, variance_avg




def complete_assignment():
    bias = []
    variance = []
    for n in n_vals:
        curr_bias, curr_variance = calculate_bias_variance_of_n(n)
        bias.append(curr_bias)
        variance.append(curr_variance)
        print("completed for n value: " + str(n))
    bias_square = []
    for value in bias:
        bias_square.append(value*value)

    # plot graph for n vs bias and n vs variance
    # and
    # plot graph for n vs bias^2 and n vs variance

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("bias - variance and bias^2 - variance")

    ax1.plot(n_vals, bias, label = "bias err")
    ax1.plot(n_vals, variance, label = "variance err")
    ax1.legend()
    plt.xlabel('n value')
    plt.ylabel('error')
    
    ax2.plot(n_vals, bias_square, label = "bias^2 err")
    ax2.plot(n_vals, variance, label = "variance err")
    ax2.legend()
    plt.xlabel('n value')
    plt.ylabel('error')

    plt.show()


complete_assignment()

# testing ---
# print(type(pts))
# print(pts[0][0], pts[0][1])