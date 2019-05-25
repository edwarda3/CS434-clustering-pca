import numpy

from parser import read_data
import kmeans


if __name__ == "__main__":
    import sys
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("data",help="Data to cluster")
    argparser.add_argument("min_k",help="Minimum K for K-Means clustering")
    argparser.add_argument("max_k",help="Maximum K for K-Means clustering")
    args = argparser.parse_args()

    #Get the ranges of K to test for.
    min_k = int(args.min_k)
    max_k = int(args.max_k)
    if(min_k >= max_k):
        print("Minimum K must be smaller than Maximum K!",file=sys.stderr)

    data = read_data(args.data)

    #Run each k min_finding_iterations times. Take the best SSE and then plot it.
    best_sses = [0]*(max_k + 1)
    min_finding_iterations = 4
    for k in range(min_k,max_k+1):
        min_sse = -1
        for i in range(min_finding_iterations):
            print("Running k={}, Attempt {}".format(k,i+1))
            means, sses = kmeans.find_means(data,k,sse_print=False)
            this_sse = sses[-1]
            print("\tSSE for this attempt: {}".format(this_sse))
            
            if(min_sse < 0 or this_sse < min_sse):
                min_sse = this_sse
        best_sses[k] = min_sse
        print("\nBest SSE for k={}: {}\n---".format(k,min_sse))

    import matplotlib.pyplot as plt
    x_axis = list(range(min_k,max_k+1))
    y_axis = best_sses[min_k:]
    plt.plot(x_axis,y_axis)
    plt.title('Total SSE for different K')
    plt.xlabel('K used for clustering')
    plt.ylabel('Sum of Squared Error')
    plt.tight_layout()
    plt.savefig('diff_k_{}_to_{}.png'.format(min_k,max_k))
    plt.show()