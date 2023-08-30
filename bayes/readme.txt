
=========================================================================

We have created two load_data functions, one for iris data and another for OCR data - because they both are in different format

first I have created a function to load the classes and it's probability of occurance - get_classes_of_array() function

next I have created a main function for naive bayes classifier, there it is calculating the classes(probability of classes) and the populate_dictionary() function =>
     which will calculate all the occurances of classes and their attributes
     it is in the format of dict(dict(dict(classes) -> count))
     classes count for every attribute in iris data, goes below ->>
    {
    0: {
        'Iris-virginica': 36,
        'Iris-setosa': 40,
        'Iris-versicolor': 44
    },
    1: {
        'Iris-virginica': 36,
        'Iris-setosa': 40,
        'Iris-versicolor': 44
    },
    2: {
        'Iris-virginica': 36,
        'Iris-setosa': 40,
        'Iris-versicolor': 44
    },
    3: {
        'Iris-virginica': 36,
        'Iris-setosa': 40,
        'Iris-versicolor': 44
    }
    }
    classes and their probability with respect to the unique attribute values ->>
    {
    0: {
        7: {
            'Iris-virginica': 0.3055555555555556,
            'Iris-versicolor': 0.1590909090909091
        },
        6: {
            'Iris-virginica': 0.5555555555555556,
            'Iris-versicolor': 0.75,
            'Iris-setosa': 0.075
        },
        5: {
            'Iris-setosa': 0.825,
            'Iris-versicolor': 0.09090909090909091
        },
        8: {
            'Iris-virginica': 0.1388888888888889
        },
        4: {
            'Iris-setosa': 0.1
        }
    },
    1: {
        3: {
            'Iris-virginica': 0.8333333333333334,
            'Iris-setosa': 0.55,
            'Iris-versicolor': 0.75
        },
        2: {
            'Iris-versicolor': 0.25,
            'Iris-virginica': 0.08333333333333333,
            'Iris-setosa': 0.025
        },
        4: {
            'Iris-setosa': 0.425,
            'Iris-virginica': 0.08333333333333333
        }
    },
    2: {
        6: {
            'Iris-virginica': 0.4722222222222222
        },
        2: {
            'Iris-setosa': 0.55
        },
        3: {
            'Iris-versicolor': 0.045454545454545456
        },
        1: {
            'Iris-setosa': 0.45
        },
        5: {
            'Iris-virginica': 0.4444444444444444,
            'Iris-versicolor': 0.29545454545454547
        },
        4: {
            'Iris-versicolor': 0.6590909090909091
        },
        7: {
            'Iris-virginica': 0.08333333333333333
        }
    },
    3: {
        2: {
            'Iris-virginica': 0.9722222222222222,
            'Iris-versicolor': 0.29545454545454547
        },
        0: {
            'Iris-setosa': 0.975
        },
        1: {
            'Iris-versicolor': 0.7045454545454546,
            'Iris-setosa': 0.025,
            'Iris-virginica': 0.027777777777777776
        }
    }
    }
    

    this way we got the probabilities of attributes, with respect to the classes

Next I have implemented, a function to predict the class based on the given input -->> calculate_class_bayes() function 

NExt I have implemented, a function to calculate the accuracy for the given test data, 


===============================================================