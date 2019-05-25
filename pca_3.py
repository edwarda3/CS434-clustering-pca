import numpy
import math
import warnings

warnings.filterwarnings('ignore')

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
    print('Finding Covariance matrix...(0/{})'.format(data.shape[0]),end='\r')
    for i in range(data.shape[0]):
        data_instance = data[i,:]
        part = data_instance-center
        new_part = numpy.dot(part.reshape(part.shape[0],1),part.reshape(1,part.shape[0]))
        c_mat = c_mat + new_part 
        print('Finding Covariance matrix...({}/{})'.format(i+1,data.shape[0]),end='\r')
    print()
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

#Using the eigenvalues, take the corresponding eigenvector and multiply each data instance with it, to get a single product. This is a new feature for that data instance. For each instance, do this target_dim times to get target_dim features for each data point.
# Importantly, this new representation keeps the same indexes as the original dataset.
def reduce_dim(data,eigens,target_dim):
    if(target_dim > len(eigens)):
        print("Target dimension must be lower than the original vector size")
        exit(1)
    #set up eigen vectors so we dont have to fetch every time
    eigen_vectors = numpy.zeros((target_dim,data.shape[1]))
    for i in range(target_dim):
        (_,e_vec) = eigens[i]
        eigen_vectors[i,:] = e_vec

    v_dim = data.shape[1]
    new_repr = numpy.zeros((data.shape[0],target_dim))
    for instance_idx in range(data.shape[0]):
        data_instance = data[instance_idx,:]
        for e_vec_i in range(target_dim):
            e_vec = eigen_vectors[e_vec_i,:]
            new_repr[instance_idx,e_vec_i] = numpy.dot(e_vec.reshape(1,v_dim),data_instance.reshape(v_dim,1))
        print("Calculating reduced features...({}/{})".format(instance_idx+1,data.shape[0]),end='\r')
    print()
    return new_repr

#For each new feature in the reduced featureset, find the data point with the highest value in that feature. Then, place the original image of that data point next to the image of the eigenvector used to make that feature.
def compare_eigen_image_with_largest_in_dim(eigens,data,original_data):
    original_dim = original_data.shape[1]
    num_to_find = data.shape[1]

    #Get the list of eigenvectors that we will use
    eigen_vectors = numpy.zeros((num_to_find,original_dim))
    for i in range(num_to_find):
        (_,e_vec) = eigens[i]
        max_e = numpy.amax(e_vec)
        for j in range(e_vec.shape[0]):
            color = e_vec[j].real/max_e
            #color = max(min(color,1.),0.)
            eigen_vectors[i,j] = color

    #Get the list of data points we will use. We find the max of each column then take the data point at that index.
    max_data_vectors = numpy.zeros((num_to_find,original_dim))
    for col in range(num_to_find):
        instance_with_max = data[:,col].argmax()
        raw_instance = original_data[instance_with_max,:]
        instance_max = numpy.amax(raw_instance)
        max_data_vectors[col,:] = raw_instance/instance_max

    

    """ for i in range(original_dim):
        print("({:.2f} | {:.2f})\t".format(eigen_vectors[0,i],max_data_vectors[0,i]),end='')
    print('\n---') """

    #We place the eigen image and the data image next to each other in a matrix which represents our pixels.
    image_single_dim = math.ceil(math.sqrt(original_dim))
    image_set = numpy.ones((image_single_dim*num_to_find, image_single_dim*2))
    for i in range(num_to_find):
        draw_row = i * image_single_dim
        draw_col_start = 0
        draw_col_mid = image_single_dim

        image_set[draw_row : draw_row+image_single_dim, draw_col_start:draw_col_start+image_single_dim] = eigen_vectors[i,:].reshape((image_single_dim,image_single_dim))
        image_set[draw_row : draw_row+image_single_dim, draw_col_mid:draw_col_mid+image_single_dim] = max_data_vectors[i,:].reshape((image_single_dim,image_single_dim))
    import matplotlib.pyplot as plt
    plt.imshow(image_set,cmap='pink')
    plt.colorbar()
    plt.axis('off')
    plt.title('Comparison of High value images\n to corresponsing eigenvectors\n Left: Eigenvector Image, \nRight: Data Image with High value')
    plt.tight_layout()
    plt.savefig("eigen_and_max_compared_{}_dims.png".format(num_to_find))
    #plt.show()



if __name__ == "__main__":
    data_file = 'p4-data.txt'
    data = read_data(data_file)
    print("data shape: {}".format(data.shape))

    cov_matrix = get_covariance_matrix(data)

    eigens = get_eigenvalues(cov_matrix)

    #show_eigen_vectors(eigens,10,int(pow(data.shape[1],.5)))

    reduced_repr = reduce_dim(data,eigens,10)
    print("reduced shape: {}".format(reduced_repr.shape))

    compare_eigen_image_with_largest_in_dim(eigens,reduced_repr,data)