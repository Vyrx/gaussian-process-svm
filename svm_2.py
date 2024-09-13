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

    C_values = [0.01 * 10 ** i for i in range(4)]
    best_acc = 0
    best_param_linear = 1

    ## Linear
    for c in C_values:
        acc = svm.svm_train(y_train, x_train, f"-s 0 -t 0 -v 5 -c {c}")

        if acc > best_acc:
            best_acc = acc
            best_param_linear = c


    ## Poly
    degree_values = [i + 1 for i in range(3)]
    best_acc = 0
    best_param_poly = []

    for c in C_values:
        for deg in degree_values:
            acc = svm.svm_train(y_train, x_train, f"-s 0 -t 1 -v 5 -c {c} -d {deg}")

            if acc > best_acc:
                best_acc = acc
                best_param_poly = [c, deg]

    ## RBF
    gamma_values = np.logspace(-2, 2, num=4)
    best_acc = 0
    best_param_rbf = []

    for c in C_values:
        for gamma in gamma_values:
            acc = svm.svm_train(y_train, x_train, f"-s 0 -t 2 -v 5 -c {c} -g {gamma}")

            if acc > best_acc:
                best_acc = acc
                best_param_rbf = [c, gamma]

    ## Train our models with the best parameter

    model_linear = svm.svm_train(y_train, x_train, f"-s 0 -t 0 -c {best_param_linear}")
    result_linear = svm.svm_predict(y_test, x_test, model_linear)
    model_poly = svm.svm_train(y_train, x_train, f"-s 0 -t 1 -c {best_param_poly[0]} -d {best_param_poly[1]}")
    result_poly = svm.svm_predict(y_test, x_test, model_poly)
    model_rbf = svm.svm_train(y_train, x_train, f"-s 0 -t 2 -c {best_param_rbf[0]} -g {best_param_rbf[1]}")
    result_rbf = svm.svm_predict(y_test, x_test, model_rbf)

    print("---Parameters for each kernel that give the best accuracy---")
    print(f"Linear; c = {best_param_linear}")
    print(f"Polynomial; [c, degree] = [{best_param_poly[0]}, {best_param_poly[1]}]")
    print(f"RBF; [c, gamma] = [{best_param_rbf[0]}, {best_param_rbf[1]}]")

    print("---Accuracy for each kernel---")
    print(f"Linear: {result_linear[1][0]}%")
    print(f"Polynomial: {result_poly[1][0]}%")
    print(f"RBF: {result_rbf[1][0]}%")

    
    X_test_file.close()
    X_train_file.close()
    Y_test_file.close()
    Y_train_file.close()