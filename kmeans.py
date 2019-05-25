import numpy
import random

from parser import read_data

ITERS = 200

#We use the class Center to represent each cluster.
#center is the vector which represents the center point.
class Center:
    def __init__(self,center,dim):
        self.center = center
        self.dim = dim
        self.community = []

    def purge_community(self):
        self.community = []

    def recompute_center(self):
        center = numpy.array([0]*self.dim)
        for member in self.community:
            center = center + member
        center = center / len(self.community)
        self.center = center

    def get_sse(self):
        sse = 0
        for member in self.community:
            sse += numpy.linalg.norm(self.center - member)
        return sse

    def center_str(self):
        return '<{}>'.format(' '.join(['{:.2f}'.format(i) for i in self.center]))

    def __str__(self):
        return "Center with community of {} members. SSE: {}".format(len(self.community), self.get_sse())

#Find the centers over a fixed number of iterations. The iterations will stop once the SSE stops changing, as that means we have converged.
def find_means(data,k,sse_print=True):
    data_dim = data.shape[0]
    k_centers = [Center(data[random.randint(0,data_dim-1),:],data.shape[1]) for _ in range(k)]
    sses = []
    
    for it in range(ITERS):
        #Reset each center community for reassignment.
        for center in k_centers:
            center.purge_community()

        #Find the closest center and migrate to its community.
        for data_idx in range(data_dim):
            closest_dist,closest_center = -1,-1
            for center in k_centers:
                this_dist = numpy.linalg.norm(data[data_idx,:] - center.center)
                if(this_dist < closest_dist or closest_dist==-1):
                    closest_dist = this_dist
                    closest_center = center
            closest_center.community.append(data[data_idx,:])
        
        #Recompute each center's center point and get its SSE.
        total_sse = 0
        for center in k_centers:
            center.recompute_center()
            total_sse += center.get_sse()

        if(sse_print):
            print("SSE, iteration {}: {}".format(it,total_sse))
        #If our sse didn't change, break the loop as we have converged.
        if(sses and total_sse == sses[-1]): 
            break
        sses.append(total_sse)

    return k_centers, sses

#Plots the TOTAL SSE of all clusters over iterations.
def make_plot(sses):
    import matplotlib.pyplot as plt
    plt.plot(list(range(1,len(sses)+1)),sses)
    plt.title('Sum of SSE of all Clusters over iterations ')
    plt.xlabel('Iterations')
    plt.ylabel('Total Sum of SSE')
    plt.tight_layout()
    plt.savefig('kmeans.png')
    plt.show()

if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("data",help="Data to cluster")
    argparser.add_argument("k",help="K for K-Means clustering")
    args = argparser.parse_args()

    data = read_data(args.data)

    means, sses = find_means(data,int(args.k))

    for mean in means:
        print(mean)

    make_plot(sses)