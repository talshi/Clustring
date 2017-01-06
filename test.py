
import kmeans

vectors = [
    [1,2],
    [2,3],
    [3,4],
    [10,15],
    [8,6]
]

km = kmeans.KMeans(k=2, vectors=vectors)
km.main_loop()