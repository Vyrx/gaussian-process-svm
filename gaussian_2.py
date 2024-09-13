import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

def rational_quadratic_kernel(data_1, data_2, kernel_variance, alpha, l):
    return kernel_variance * pow( 1 + ( (data_1 - data_2) ** 2 ) / (2 * alpha * l * l), (-1) * alpha )

def log_likelihood(theta):
    alpha = theta[0]
    l = theta[1]
    kernel_variance = theta[2]
    y = np.array([np.transpose(data)[1]]).T
    cov = []

    ## Construct covariance function

    for i in range(data_count):
        temp = []
        for j in range(data_count):

            result = rational_quadratic_kernel(data[i][0], data[j][0], kernel_variance, alpha, l)

            if i == j:
                result += 1 / beta
            
            temp.append(result)
        cov.append(temp)
    
    cov_inv = np.linalg.inv(cov)
    cov_det = np.linalg.det(cov)

    result = 0.5 * np.log(cov_det) + 0.5 * np.matmul(np.matmul(y.T, cov_inv), y) + (data_count/2) * np.log(2 * np.pi)
    result = result[0][0]
    return result

if __name__ == "__main__":
    file = open("data/input.data", "r")
    global data 
    data = []


    for line in file:
        temp = []
        temp.append(float(line.split(" ")[0]))
        temp.append(float(line.split(" ")[1]))
        data.append(temp)

    
    beta = 5
    cov = []
    kernel = []
    global data_count
    data_count = len(data)
    
    ## Parameters
    alpha = 1
    l = 5
    kernel_variance = 1

    result = minimize(log_likelihood, [100, 5, 1], method = 'Nelder-Mead', options={"disp": True})
    if result.success:
        print(result.x)
        alpha = result.x[0]
        l = result.x[1]
        kernel_variance = result.x[2]
    else:
        print("Can't find result")

    ## Construct covariance function
    for i in range(data_count):
        temp = []
        for j in range(data_count):

            result = rational_quadratic_kernel(data[i][0], data[j][0], kernel_variance, alpha, l)

            if i == j:
                result += 1 / beta
            
            temp.append(result)
        cov.append(temp)   

    cov_inv = np.linalg.inv(cov)
    mean = []
    var = []

    x = np.linspace(-60,60, 500)

    ## Get mean and variance of points in range [-60, 60]
    for item in x:
        temp_kern = [rational_quadratic_kernel(temp, item, kernel_variance, alpha, l) for temp in np.transpose(data)[0]]
        temp_kern = np.array([temp_kern]).T

        temp_mean = np.matmul(np.matmul(temp_kern.T, cov_inv), np.transpose(data)[1])[0]
        temp_var = rational_quadratic_kernel(item, item, kernel_variance, alpha, l) + 1 / beta - np.matmul(np.matmul(temp_kern.T, cov_inv), temp_kern)
        temp_var = temp_var[0][0]
        
        mean.append(temp_mean)
        var.append(temp_var)

    plt.plot(x, mean, 'b')
    plt.fill_between(x, [mean[i] + np.sqrt(1.96 * var[i]) for i in range(len(x))], [mean[i] - 1.96 * np.sqrt(var[i]) for i in range(len(x))], color = 'lightblue')    
    plt.scatter(np.transpose(data)[0], np.transpose(data)[1], color = 'black')
    plt.show()


    file.close()