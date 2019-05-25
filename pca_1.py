import numpy

from parser import read_data

#Computes the average vector of all points by adding them and them dividing by number of points.
def get_data_center(data):
    center = numpy.zeros(data.shape[1])
    for i in range(data.shape[0]): #for each row
        center = center + data[i,:]
    center = center / data.shape[0] #divide by num rows
    return center

#Sum the matrices of each point minus the center minus the transpose of it. Then divide by n.
# (1/n)*sum((xi-u) * (xi-u).T)
def get_covariance_matrix(data):
    center = get_data_center(data)
    
    c_mat = numpy.zeros((data.shape[1],data.shape[1]))
    #print("cov_shape: {}".format(c_mat.shape))
    for i in range(data.shape[0]):
        data_instance = data[i,:]
        part = data_instance-center
        new_part = numpy.dot(part.reshape(part.shape[0],1),part.reshape(1,part.shape[0]))
        c_mat = c_mat + new_part 
    c_mat = c_mat/data.shape[0]
    
    return c_mat

#Get the eigenvalues, and pair them with the corresponding eigenvectors. Then sort this list in descending order for eigenvalues. 
def get_eigenvalues(matrix):
    e_values, e_vectors = numpy.linalg.eig(matrix)
    eigens = []
    for i in range(e_values.shape[0]):
        eigens.append((e_values[i],e_vectors[:,i]))
    eigens.sort(key=lambda x:x[0],reverse=True)

    return eigens


if __name__ == "__main__":
    data_file = 'p4-data.txt'
    data = read_data(data_file)
    print("data shape: {}".format(data.shape))

    cov_matrix = get_covariance_matrix(data)

    eigens = get_eigenvalues(cov_matrix)

    num_to_show = 10
    print("Top 10 Eigen Values in decreasing order:")
    for i in range(num_to_show):
        (e_val,_) = eigens[i]
        print(e_val.real)