import numpy
import math

from parser import read_data

#Computes the average vector of all points by adding them and them dividing by number of points.
def get_data_center(data):
    center = numpy.zeros(data.shape[1])
    for i in range(data.shape[0]): #for each row
        center = center + data[i,:]
    center = center / data.shape[0] #divide by num rows
    return center

def show_mean(mean):
    import matplotlib.pyplot as plt
    dim = math.ceil(math.sqrt(mean.shape[0]))
    image = mean.reshape((dim,dim))
    plt.imshow(image,cmap='pink')
    plt.colorbar()
    plt.axis('off')
    plt.title('Mean image from data')
    plt.savefig('mean_image.png')
    plt.clf()

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
    
    return c_mat, center

#Get the eigenvalues, and pair them with the corresponding eigenvectors. Then sort this list in descending order for eigenvalues. 
def get_eigenvalues(matrix):
    e_values, e_vectors = numpy.linalg.eig(matrix)
    eigens = []
    for i in range(e_values.shape[0]):
        eigens.append((e_values[i],e_vectors[:,i]))
    eigens.sort(key=lambda x:x[0],reverse=True)

    return eigens

#Show a image with num_to_show subimages. Each subimage is a eigenvector that is normalized to be within 0.0, 1.0. 
# We limit to 3 subimages per row.
def show_eigen_vectors(eigens,num_to_show,single_dim):
    max_cols = 4
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    setOfImages = numpy.zeros((single_dim*math.ceil(num_to_show/max_cols),single_dim*max_cols))
    for e in range(num_to_show):
        (_,e_vec) = eigens[e]
        e_img = e_vec.reshape((single_dim,single_dim))
        max_e = numpy.amax(e_img)
        e_formatted_img = numpy.zeros((single_dim,single_dim))
        for i in range(e_img.shape[0]):
            for j in range(e_img.shape[1]):
                color = e_img[i,j]
                """ color = 1.4 * e_img[i,j].real/max_e
                color = max(min(color,1.),0.) """
                e_formatted_img[i,j] = color
                #print("{:.2f} ".format(e_formatted_img[i,j]),end='')
            #print()
        colNum = (e%max_cols) * single_dim
        rowNum = (e//max_cols) * single_dim
        setOfImages[rowNum:rowNum+single_dim,colNum:colNum+single_dim] = e_formatted_img
    
    plt.imshow(setOfImages,cmap='pink')
    plt.colorbar()
    plt.axis('off')
    plt.title('Eigenvectors of highest eigenvalues\nhigh: top-left, stepping right.')
    plt.savefig("eigen_images_top_{}.png".format(num_to_show))
    #plt.show()

if __name__ == "__main__":
    data_file = 'p4-data.txt'
    data = read_data(data_file)
    print("data shape: {}".format(data.shape))

    cov_matrix, mean_vector = get_covariance_matrix(data)
    show_mean(mean_vector)

    eigens = get_eigenvalues(cov_matrix)

    show_eigen_vectors(eigens,10,int(pow(data.shape[1],.5)))