#ifndef _KERNEL_HH
#define _KERNEL_HH

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

#define COPUTER_NUMBER_OF_THREADS 8
#define PI 3.141592
#define MPI_MINIMUM_DISTANCE_COMPUTER 1
#define MPI_NOT_MINIMUM_DISTANCE_COMPUTER 0

struct point_t
{
	int id;// point id
	float a; //circle x coordinate
	float b; //circle y coordinate
	float r; //point radius distance from circle coordinate
	float x; //point x coordinate - changes as dt change
	float y; //point y coordinate - changes as dt change
}typedef Point;

struct cluster_t
{
	int id; // cluster id
	float x; //cluster x coordinate
	float y; //cluster y coordinate
	float sumX; // sum all x coordinates from all related points
	float sumy; // sum all y coordinates from all related points
	int pointsCounter; // count the number of points that related to this cluster

}typedef Cluster;
int minimumDistanceComputerID(float* distanceArray , int size);
void comparingValuesWithOtherComputers(Cluster* minimumDistanceClustersArray ,float* minimumDistance , float* minimumDistanceDT , int numberOfClusers,int myid , int numprocs , MPI_Datatype& clusterType); //each slave send to the master it's minimum distance - after master compare all result's and gathering the information (if needed) from slaves.
float calculateMinimumClustersDistance(Cluster* clustersArray , int numberOfClusers);//return the minimum distance between clusters from the cluster at clustersArray.
bool compareClusters(Cluster& clusters1 , Cluster& clusters2);//returning true if x and y are equals.
bool isClustersArrayEquals(Cluster* clustersArray1 , Cluster* clustersArray2 , int numberOfClusers);//return true is all clusters at same index are the equals. else return false.
void resetClustersValues(Cluster* clustersArray , int numberOfClusers);//setting each cluster sumX,sumy and pointsCounter values to zero.
void copyClusters(Cluster* destinationClusterArray , Cluster* sourceClusterArray , int numberOfClusers);//copy the data in sourceClusterArray to destinationClusterArray.
void calculeteClustersNewCoordinate(Cluster* clustersArray , int numberOfClusers);//calculete the new (x,y) coordinate of each cluster based on sumX, sumy and pointsCounter.
void minDistanceBetweenPointToCluster(Point& point , Cluster* clustersArray , int numberOfClusers , Cluster* threadClusterArray); //finding the shortest distance between the point and a cluster from the clustersArray , and adding the point to the threadClusterArray - the array of the thread that called this function.
void k_means(Point* pointsArray , int numberOfPoints , Cluster** threadsClusterMatrix ,Cluster* clusterArray , int numberOfClusers); // calculetiing new clusters arrays and set it in clusterArray var.
void setAllThreadsClustersArrays(Cluster** clusterMatrix , int numOfClusters); //setting each thread - cluster array.
void setClusters(Point* pointsArray , Cluster* clusterArray , int numberOfClusters); //set the first numberOfClusters point from pointsArray to be clusters. init other values to zero
void mainAlgoritem(Cluster* minimumDistanceClustersArray ,float* minimumDistance , float* minimumDistanceDT,Point* pointsArray ,int numberOfPoints ,int numberOfClusers ,float dt ,float maxT ,int myid, int numprocs,int numberOfIteration);
void brodcastParameters(int* numberOfPoints ,Point* pointsArray ,int* numberOfClusers ,float* dt ,float* maxT , int* numberOfIteration , int myid , MPI_Datatype& typePoint); // brodcast all parameters from master to all slaves. brodcast know who is slave and who is master.
void creatMpiPointStruct(MPI_Datatype& pointType); // create a point data type for mpi.
void creatMpiClusterStruct(MPI_Datatype& clusterType); // create a cluster data type for mpi.
void Master(int myid , int numprocs); //master code
void Slave(int myid , int numprocs); // slave code
void writeToFile(int numberOfClusters , Cluster* minimumDistanceClustersArray , float minimumDistance , float minimumDistanceDT , float run_time , char* writeFileName); //writing all releavent parameters on output files.
Point* readFromFile(int* numberOfPoints ,int* numberOfClusers , float* dt , float* maxT , int* numberOfIteration,char* readFileName);// init all function signature variable and return initiated Points array.
float distanceBetweenPointAndCluster(Point& point, Cluster& cluster); //return the distance between the point and the cluster.
float distanceBetweenClusters(Cluster& clusterOne, Cluster& clusterTwo); //return the distance between the clusters.
///cuda
void initCodaDevice(); //initiating coda device
void copyPointArrayToCudaDevice(Point* pointsArray , int size); // copying points array to cuda memory.
void movePointsLocationWithCuda(Point* pointsArray , float dt, float maxT , float pi); // calculating new points coordinates on cuda device.
void releaseCudaMemory(); //releasing memory on coda device

#endif
