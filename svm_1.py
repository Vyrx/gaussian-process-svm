import numpy as np
import libsvm.svmutil as svm

def process_csv(file):
    result = []

    for line in file:
        temp = line.split(",")
        result.append([float(item) for item in temp])
    
    return result


if __name__ == "__main__":
    X_test_file = open("data/X_test.csv", "r")
    X_train_file = open("data/X_train.csv", "r")
    Y_test_file = open("data/Y_test.csv", "r")
    Y_train_file = open("data/Y_train.csv", "r")

    x_train = process_csv(X_train_file)
    y_train = [int(item) - 1 for item in Y_train_file]
    x_test = process_csv(X_test_file)
    y_test = [int(item) - 1 for item in Y_test_file]

    model_linear = svm.svm_train(y_train, x_train, "-s 0 -t 0")
    result_linear = svm.svm_predict(y_test, x_test, model_linear)
    model_poly = svm.svm_train(y_train, x_train, "-s 0 -t 1")
    result_poly = svm.svm_predict(y_test, x_test, model_poly)
    model_rbf = svm.svm_train(y_train, x_train, "-s 0 -t 2")
    result_rbf = svm.svm_predict(y_test, x_test, model_rbf)

    print("---Accuracy for each kernel---")
    print(f"Linear: {result_linear[1][0]}%")
    print(f"Polynomial: {result_poly[1][0]}%")
    print(f"RBF: {result_rbf[1][0]}%")

    X_test_file.close()
    X_train_file.close()
    Y_test_file.close()
    Y_train_file.close()