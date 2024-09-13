import matplotlib.pyplot as plt
import numpy as np

def rational_quadratic_kernel(data_1, data_2, kernel_variance, alpha, l):
    return kernel_variance * pow( 1 + ( (data_1 - data_2) ** 2 ) / (2 * alpha * l * l), (-1) * alpha )

if __name__ == "__main__":
    file = open("data/input.data", "r")
    data = []


    for line in file:
        temp = []
        temp.append(float(line.split(" ")[0]))
        temp.append(float(line.split(" ")[1]))
        data.append(temp)


    beta = 5
    cov = []
    data_count = len(data)

    ## Parameters
    alpha = 1
    l = 5
    kernel_variance = 1

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
    plt.fill_between(x, [mean[i] + 1.96 * np.sqrt(var[i]) for i in range(len(x))], [mean[i] - 1.96 * np.sqrt(var[i]) for i in range(len(x))], color = 'lightblue')    
    plt.scatter(np.transpose(data)[0], np.transpose(data)[1], color = 'black')
    plt.show()


    file.close()