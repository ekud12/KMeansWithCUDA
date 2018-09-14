#include "kernel.h"
#include <time.h>
#include <omp.h>
#include <fstream>
#include <iostream>

using namespace std;

/****************************************************************************************************************
Program's Main - initialize MPI and CUDA device and 
****************************************************************************************************************/
int main(int argc , char *argv[])
{
	int myid,numprocs;
	MPI_Init(&argc , &argv);
	initCodaDevice();

	MPI_Comm_rank(MPI_COMM_WORLD ,&myid);

	MPI_Comm_size(MPI_COMM_WORLD,&numprocs);

	if(myid == 0)
	{
		Master(myid,numprocs);
	}
	else
	{
		Slave(myid,numprocs);
	}

	//We finalize our program
	MPI_Finalize();
}

/****************************************************************************************************************
Function finds and return the ID of the proccessor with the minimum distance cluster
****************************************************************************************************************/
int minimumDistanceComputerID(float* distanceArray , int size)
{
	int tempMinIndex;
	for(int i = 0 ; i < size ; i++)
		for(int j = i+1 ; j < size ; j++)
		{
			if(distanceArray[i] < distanceArray[j])
				tempMinIndex = i;
			else
				tempMinIndex = j;
		}
	return tempMinIndex;
}

/****************************************************************************************************************
Function finds and return the ID of the proccessor with the minimum distance found between 2 clusters in its domain
We basically want to find out which computer "Won" by having the shortest distance between 2 clusters and get it's
DT at the moment of finding out the minimum distance and clusterArray it holds (We want to save to file at the end)
- minimumDistanceClustersArray - 
****************************************************************************************************************/
void comparingValuesWithOtherComputers(Cluster* minimumDistanceClustersArray ,float* minimumDistance , float* minimumDistanceDT , int numberOfClusers,int myid , int numprocs , MPI_Datatype& clusterType)
{
	MPI_Status status;
	//Master performs this part
	if(myid == 0)
	{
		//init vars
		int flag_min = MPI_MINIMUM_DISTANCE_COMPUTER , flag_not_min = MPI_NOT_MINIMUM_DISTANCE_COMPUTER;
		int i,computer_id,minDistanceComputerNumber;
		float distance_from_slave;
		//allocate memory for array according to how many proccessors we have
		float* minDistanceArrayFromComputer = (float*)malloc(numprocs * sizeof(float));
		//we set the first computer as the winner in the race to find the minimum distance
		minDistanceArrayFromComputer[0] = *minimumDistance;

		//we go over all of proccessors/computers, the master recieves the minimum distances and id from each computer and fills it's 
		//minDistanceArrayFromComputer elements according to data returned from all the slaves
		for( i = 1 ; i < numprocs ; i++)
		{
			MPI_Recv(&distance_from_slave,1,MPI_FLOAT,MPI_ANY_SOURCE,0,MPI_COMM_WORLD,&status);
			computer_id = status.MPI_SOURCE;
			minDistanceArrayFromComputer[computer_id] = distance_from_slave;
		}

		//We call the method minimumDistanceComputerID(), in order to find the computer that holds the winning minimun
		//distance between 2 clusters
		minDistanceComputerNumber = minimumDistanceComputerID(minDistanceArrayFromComputer,numprocs);

		//After we have the winner, the Master sends notification to all the other slaves that they are not 
		//the minimum using MPI_NOT_MINIMUM_DISTANCE_COMPUTER flag.
		for( i=1 ; i< numprocs ; i++)
			if(i != minDistanceComputerNumber)
				MPI_Send(&flag_not_min,1,MPI_INT,i,0,MPI_COMM_WORLD); 

		//if the Master doesnt hold the minimum distance, hence not the winner, it has to get the details from the winning slave,
		//so we send to the computerID a notification he holds the minimum winning distance, and we get from him his:
		// - minimumDistanceDT, -minimumDistanceClustersArray, and set the global minimumDistance to his distance.
		if(minDistanceComputerNumber !=0)
		{
			MPI_Send(&flag_min,1,MPI_INT,minDistanceComputerNumber,0,MPI_COMM_WORLD);

			*minimumDistance = minDistanceArrayFromComputer[minDistanceComputerNumber];
			MPI_Recv(minimumDistanceDT , 1 , MPI_FLOAT , minDistanceComputerNumber , 0 , MPI_COMM_WORLD,&status);
			MPI_Recv(minimumDistanceClustersArray , numberOfClusers , clusterType , minDistanceComputerNumber , 0 , MPI_COMM_WORLD,&status);
		}
		//we free our array
		free(minDistanceArrayFromComputer);
	}
	//if we are the slave, we just send our minimum distance, get answer from master if we won or not
	//and if true, we send our minimumDistanceDT and minimumDistanceClustersArray.
	else
	{
		MPI_Send(minimumDistance,1,MPI_FLOAT,0,0,MPI_COMM_WORLD);
		int master_answer;
		MPI_Recv(&master_answer,1,MPI_INT,0,0,MPI_COMM_WORLD,&status); 
		if(master_answer == MPI_MINIMUM_DISTANCE_COMPUTER)
		{
			MPI_Send(minimumDistanceDT,1,MPI_FLOAT,0,0,MPI_COMM_WORLD);
			MPI_Send(minimumDistanceClustersArray,numberOfClusers,clusterType,0,0,MPI_COMM_WORLD);
		}
	}
}

/****************************************************************************************************************
Function calculate the minimum distance between any 2 clusters in the clustersArray
****************************************************************************************************************/
float calculateMinimumClustersDistance(Cluster* clustersArray , int numberOfClusers)
{
	float minimum = -100;
	//setting a distance array as the number of proccessors
	float distanceArray[COPUTER_NUMBER_OF_THREADS];
	int i,tid;

	//initializing the distance array on all proccessors to -100
	#pragma omp for schedule(dynamic) private(i)
		for(i=0 ; i< COPUTER_NUMBER_OF_THREADS ;i++)
			distanceArray[i] = -100;

	//Each proccessor calculate it's own distances between all of it's clusters
	#pragma omp parallel private(i,tid)
	{
		 tid = omp_get_thread_num();
		for(i = 0  ; i < numberOfClusers ; i++)
			for(int j = i+1 ; j < numberOfClusers ; j++)
			{
				//if we hadnt calculated yet, just get the distance
				if(distanceArray[tid] == -100)
					distanceArray[tid] = distanceBetweenClusters(clustersArray[i],clustersArray[j]);
				else
				{
					//if we have previous result, check if the new distance is smaller, if true than
					//set the distance to the new result
					float temp = distanceBetweenClusters(clustersArray[i],clustersArray[j]);
					if(temp < distanceArray[tid])
						distanceArray[tid] = temp;
				}
			}
	//wait for all proccessors to check their distances
	#pragma omp barrier
	}

	//run on all the distances and find the smallest, and keep it in minimum variable
	for(i=0 ; i < COPUTER_NUMBER_OF_THREADS ; i++)
		if(minimum == -100)
			minimum = distanceArray[i];
		else
			if(distanceArray[i] < minimum)
				minimum = distanceArray[i];

	//return the smallest distance found between all the clusters
	return minimum;
}

/****************************************************************************************************************
Function check if the two clusters centers are the same, and return true only if both are.
****************************************************************************************************************/
bool compareClusters(Cluster& clusters1 , Cluster& clusters2)
{
	if(clusters1.x != clusters2.x)
		return false;

	if(clusters1.y != clusters2.y)
		return false;

	return true;
}

/****************************************************************************************************************
Function check equality between two given clusters using the compareClusters() method.
if we had at least one x or y not equal, meaning the centers of the clusters are not the same, we return false,
true otherwise.
****************************************************************************************************************/
bool isClustersArrayEquals(Cluster* clustersArray1 , Cluster* clustersArray2 , int numberOfClusers)
{
	int i,counter = 0;
#pragma omp parallel reduction (+: counter) private(i)
{
	for(i = 0 ; i < numberOfClusers ; i++)
		if(compareClusters(clustersArray1[i] , clustersArray2[i]) == false)
			counter++;
}
	if(counter != 0)
		return false;

	return true;
}

/****************************************************************************************************************
Function resets the clusterArray's sumX sumY and pointsCounter to 0
****************************************************************************************************************/
void resetClustersValues(Cluster* clustersArray , int numberOfClusers)
{
		int i;
#pragma omp for schedule(static) private(i)
	for(i = 0; i < numberOfClusers ; i++)
	{
		clustersArray[i].sumX = 0;
		clustersArray[i].sumy = 0;
		clustersArray[i].pointsCounter = 0;
	}
}

/****************************************************************************************************************
Function copy one ClusterArray to another (DEEP copy).
****************************************************************************************************************/
void copyClusters(Cluster* destinationClusterArray , Cluster* sourceClusterArray , int numberOfClusers)
{
	int i;
#pragma omp for schedule(static) private(i)
	for(i = 0; i < numberOfClusers ; i++)
	{
		destinationClusterArray[i].id = sourceClusterArray[i].id;
		destinationClusterArray[i].pointsCounter = sourceClusterArray[i].pointsCounter ;
		destinationClusterArray[i].sumX = sourceClusterArray[i].sumX;
		destinationClusterArray[i].sumy = sourceClusterArray[i].sumy;
		destinationClusterArray[i].x = sourceClusterArray[i].x;
		destinationClusterArray[i].y = sourceClusterArray[i].y;
	}
}

/****************************************************************************************************************
Function calculate the clusterArray's new center coordinates:  the relevent Cluster's sumX and sumY divided by 
the number of points associated with the cluster.
each proccessor take part in calculating this.
****************************************************************************************************************/
void calculeteClustersNewCoordinate(Cluster* clustersArray , int numberOfClusers)
{
	int i;
#pragma omp for schedule(static) private(i)
	for(i = 0; i < numberOfClusers ; i++)
	{
		clustersArray[i].x = clustersArray[i].sumX / clustersArray[i].pointsCounter;
		clustersArray[i].y = clustersArray[i].sumy / clustersArray[i].pointsCounter;
	}
}

/****************************************************************************************************************
Function compute the minimum distance between a Single Point and all clusters centers in the specific proccessor
cluster array only!
- point - The current point
- clustersArray - The current Proccessor's clusterArray 
- numberOfClusers - The number of clusters in the afordmentioned array.
- threadClusterArray - The cluster In the Total threads clusters matrix.
****************************************************************************************************************/
void minDistanceBetweenPointToCluster(Point& point , Cluster* clustersArray , int numberOfClusers , Cluster* threadClusterArray)
{
	int cluster_id,i;
	//temps we are going to use
	float minimum_distance,temp_distance;

	// find the current minimum distance between the first cluster and the given point
	minimum_distance = distanceBetweenPointAndCluster(point,clustersArray[0]); // min distance between cluster to point
	//set the cluster_id to the first cluster id
	cluster_id = clustersArray[0].id;

	//Then, run on all clusters
	for(i=1 ; i < numberOfClusers ; i++)
	{
		//calculate the distance between the point and the current cluster
		temp_distance = distanceBetweenPointAndCluster(point,clustersArray[i]);
		//if the closest cluster is the current one replace to the current cluster_id and save the new minimum distance
		if(temp_distance < minimum_distance)
		{
			minimum_distance = temp_distance;
			cluster_id = clustersArray[i].id;
		}
	}
	
	//When we find our minimum distance between the point and the cluster we add the point to the chosen
	//cluster, increment the pointsCounter associated with the cluster and add the point's x,y values
	//to the relevent proccessor's sumX and sumY for the next interval and average calculation
	threadClusterArray[cluster_id].sumX += point.x;
	threadClusterArray[cluster_id].sumy += point.y;
	threadClusterArray[cluster_id].pointsCounter++;
}


/****************************************************************************************************************
Function compute the execute the k_means Algorithem itself.
We send to parallel threads the shared variables because they all only read them.
The tid, i column are privates because it is done per thread
****************************************************************************************************************/
void k_means(Point* pointsArray , int numberOfPoints , Cluster** threadsClusterMatrix ,Cluster* clusterArray , int numberOfClusers)
{
	int tid,i,column;

	//Starting Parallel Section
	#pragma omp parallel shared(pointsArray , threadsClusterMatrix , clusterArray) private(tid,i,column)
	{
		//Each thread get its threadID.
		tid = omp_get_thread_num(); 

	//Each thread finds the minimum distance between all the points and the chosen cluster they will be assigned to
	#pragma omp for schedule(static) private(i) 
		for( i=0  ; i < numberOfPoints ; i++)
		{
			minDistanceBetweenPointToCluster(pointsArray[i],clusterArray,numberOfClusers,threadsClusterMatrix[tid]);
		} 

	// waiting for all the proccessors to check their points
	#pragma omp barrier 

	//running on the proccessors and summing:
	//all the sumX,sumY,pointsCounter from all the first cluster in every proccessor into clusterArray[0]'s sumX,sumY,pointsCounter
	//all the sumX,sumY,pointsCounter from all the second cluster in every proccessor into clusterArray[1]'s sumX,sumY,pointsCounter
	// and so on until we finished going on all the clusters from all the proccessors
	
	#pragma omp for schedule(dynamic) private(column)
		for(column = 0 ; column < numberOfClusers ; column++)
		{
			for( int row = 0 ; row < COPUTER_NUMBER_OF_THREADS ; row++)
			{
				clusterArray[column].pointsCounter += threadsClusterMatrix[row][column].pointsCounter;
				clusterArray[column].sumX += threadsClusterMatrix[row][column].sumX;
				clusterArray[column].sumy += threadsClusterMatrix[row][column].sumy;
			}
		}
	}//pragma 

}

/****************************************************************************************************************
Function sets ALL Clusters arrays in a matrix using OMP,statically (predefined fashion for OMP distribution) 
we use this to assign the clusters to all the proccessors.
- clusterMatrix - the matrix of all the clusters from ALL the proccessors (row = proc, column = the cluster array)
- numOfClusters - the number of clusters to iterate over
****************************************************************************************************************/
void setAllThreadsClustersArrays(Cluster** clusterMatrix , int numOfClusters)
{
	int i;
#pragma omp for schedule(static) private(i)
	for( i=0 ; i < COPUTER_NUMBER_OF_THREADS ; i++)
		for(int j=0 ; j < numOfClusters ; j++)
		{
			clusterMatrix[i][j].id = j;
			clusterMatrix[i][j].pointsCounter = 0;
			clusterMatrix[i][j].sumX = 0;
			clusterMatrix[i][j].sumy = 0;
			clusterMatrix[i][j].x = 0;
			clusterMatrix[i][j].y = 0;
		}
}


/****************************************************************************************************************
Function sets Clusters using OMP, given some proccessors it run statically (predefined fashion
for OMP distribution).
we use this to select the new cluster centers arbitrarly each iteration from the points array - used on PROCCESSORS.
****************************************************************************************************************/
void setClusters(Point* pointsArray , Cluster* clusterArray , int numberOfClusters)
{
	int i;
#pragma omp for schedule(static) private(i) 
	for(i = 0 ; i < numberOfClusters ; i++)
	{
		clusterArray[i].id  = pointsArray[i].id;
		clusterArray[i].x  = pointsArray[i].x;
		clusterArray[i].y  = pointsArray[i].y;
		clusterArray[i].pointsCounter  = 0;
		clusterArray[i].sumX = 0;
		clusterArray[i].sumy = 0;
	}

}

/****************************************************************************************************************
Function that runs the main Algorithem of computation. Done by Master and Slaves.
- minimumDistanceClustersArray - the given array by the proccessor/computer that holds the clusters at minimum 
****************************************************************************************************************/
void mainAlgoritem(Cluster* minimumDistanceClustersArray ,float* minimumDistance , float* minimumDistanceDT,Point* pointsArray ,int numberOfPoints ,int numberOfClusers ,float dt ,float maxT ,int myid, int numprocs,int numberOfIteration)
{
	Cluster* currentIterationClusterArray;
	Cluster* previousIterationClusterArray; 
	Cluster** clusterMatrix;
	int i,k,compareClusterCounter;
	float currentDT,distance;

	//Efficency booleans
	bool first_iteration;
	bool first_distance = true;
	bool isEquals;
	int iterationToCheck = 3;
	
	//Allocate memory for the clusterArrays we will use to compare
	currentIterationClusterArray = (Cluster*)malloc(numberOfClusers * sizeof(Cluster));
	previousIterationClusterArray = (Cluster*)malloc(numberOfClusers * sizeof(Cluster));
	
	//We define the clusters matrix that holds clusters of all proccessors/computers and 
	//allocate memory to it
	clusterMatrix = (Cluster**)malloc(COPUTER_NUMBER_OF_THREADS * sizeof(Cluster*));
	for(i = 0 ; i < COPUTER_NUMBER_OF_THREADS ; i++)
		clusterMatrix[i] = (Cluster*)malloc(numberOfClusers * sizeof(Cluster));

	//The main runtime Loop as per DT
	for( currentDT = myid * dt ; currentDT < maxT ; currentDT = currentDT + numprocs * dt)
	{
		//Call the Cuda device and use it to move the points
		movePointsLocationWithCuda(pointsArray , currentDT , maxT , (float)PI );

		//Set new clusters according to the new points locations and the current clusterArray
		setClusters(pointsArray,currentIterationClusterArray,numberOfClusers);

		//Set init checkers
		compareClusterCounter = 0;
		first_iteration = true;
		
		//The Inner Loop as per Iterations
		for ( k = 0 ; k < numberOfIteration ; k++)
		{
			//set the Matrix Cluster
			setAllThreadsClustersArrays(clusterMatrix , numberOfClusers);

			//Run the K-Means function to calculate
			k_means(pointsArray,numberOfPoints,clusterMatrix,currentIterationClusterArray,numberOfClusers);

			//calcuting the new clusters centers coordinates.
			calculeteClustersNewCoordinate(currentIterationClusterArray , numberOfClusers);

			// not first iteration - comparing with previous clusters arrays.
			if(first_iteration == false) 
			{
				//if we recieve equality between the previous and current clustersArrays, we add to the counter of our checks,
				//if we passed the number of iterations efficency coefficent value, we set k to be numberOfIterations to exit the loop
				//else, we reset the counting
				isEquals = isClustersArrayEquals(previousIterationClusterArray , currentIterationClusterArray ,numberOfClusers );
				if (isEquals == true)
				{
					compareClusterCounter++;
					if(iterationToCheck < compareClusterCounter)
						k = numberOfIteration; 
				}
				else
					compareClusterCounter = 0;
			}
			// first iteration have no previous clusters arrays to compare with.
			else 
			{
				first_iteration = false;
			}

			///saving the current iteration cluster array in order to compare with the clusters that will be found at next iteration.
			copyClusters(previousIterationClusterArray , currentIterationClusterArray , numberOfClusers);
			
			//reseting currentIterationClusterArray values - sumX,sumy and pointsCounter.
			resetClustersValues(currentIterationClusterArray , numberOfClusers);

		}

		//first distance - comparing with previous distance.
		if(first_distance == false) 
		{
			distance = calculateMinimumClustersDistance(currentIterationClusterArray , numberOfClusers);
			if(distance < *minimumDistance)
			{
				copyClusters(minimumDistanceClustersArray , currentIterationClusterArray , numberOfClusers);
				*minimumDistance = distance;
				*minimumDistanceDT = currentDT;
			}
		}
		//first time - have no previous distance to compare with.
		else
		{
			first_distance = false;

			copyClusters(minimumDistanceClustersArray , currentIterationClusterArray , numberOfClusers);
			*minimumDistance = calculateMinimumClustersDistance(minimumDistanceClustersArray , numberOfClusers);
			*minimumDistanceDT = currentDT;
		}	
	}

	//Free all the memory allocated
	free(currentIterationClusterArray);
	free(previousIterationClusterArray);
	for( i=0 ; i < COPUTER_NUMBER_OF_THREADS ; i++)
		free(clusterMatrix[i]);
	free(clusterMatrix);


}

/****************************************************************************************************************
Function broadcasts all parameters to master+slaves on start
-numberOfPoints - how many points we going to have
-pointsArray - the points array itself (in regard to who gets its - allocation is made)
****************************************************************************************************************/
void brodcastParameters(int* numberOfPoints ,Point* pointsArray ,int* numberOfClusers ,float* dt ,float* maxT , int* numberOfIteration , int myid , MPI_Datatype& typePoint)
{
	MPI_Bcast(numberOfPoints , 1 ,MPI_INT , 0 ,MPI_COMM_WORLD);
	// if the given id is not the master than we must allocate memory for the copy of the points array.
	// if the given is the master we already allocated memory for it
	if(myid != 0 )
		pointsArray = (Point*)malloc( (*numberOfPoints) * sizeof(Point) );
	//populate the points array according to number of points
	MPI_Bcast(pointsArray , (*numberOfPoints) ,typePoint , 0 ,MPI_COMM_WORLD);
	//populate other simple variables
	MPI_Bcast(numberOfClusers , 1 ,MPI_INT , 0 ,MPI_COMM_WORLD);
	MPI_Bcast(dt , 1 ,MPI_FLOAT , 0 ,MPI_COMM_WORLD);
	MPI_Bcast(maxT , 1 ,MPI_FLOAT , 0 ,MPI_COMM_WORLD);
	MPI_Bcast(numberOfIteration , 1 ,MPI_INT , 0 ,MPI_COMM_WORLD);
}


/****************************************************************************************************************
Function creates a MPI structure datatype so MPI will recognize what a "Point" is, including all its attr:
-MPI_Datatype& pointType - the new datatype handler we create for point
****************************************************************************************************************/
void creatMpiPointStruct(MPI_Datatype& pointType)
{
	//init cluster 
	Point temp = {0,0,0,0,0,0};

	//Type of element in each block
	MPI_Datatype type[6] = {MPI_INT , MPI_FLOAT , MPI_FLOAT ,MPI_FLOAT ,MPI_FLOAT ,MPI_FLOAT};

	//Number of elements in each block
	int typeLen[6] = {1 , 1 , 1 , 1 , 1 ,1};

	//Bytes displacement in memory
	MPI_Aint typeIndices[6];
	typeIndices[0] = ( (char*)&(temp.id) ) - ( (char*)&temp );
	typeIndices[1] = ( (char*)&(temp.a) ) - ( (char*)&temp );
	typeIndices[2] = ( (char*)&(temp.b) ) - ( (char*)&temp );
	typeIndices[3] = ( (char*)&(temp.r) ) - ( (char*)&temp );
	typeIndices[4] = ( (char*)&(temp.x) ) - ( (char*)&temp );
	typeIndices[5] = ( (char*)&(temp.y) ) - ( (char*)&temp );

	//Create the type and commit so MPI now will know what the struct is.
	MPI_Type_struct(6 , typeLen, typeIndices , type , &pointType);
	MPI_Type_commit(&pointType);
}


/****************************************************************************************************************
Function creates a MPI structure datatype so MPI will recognize what a "Cluster" is, including all its attr:
-MPI_Datatype& clusterType - the new datatype handler we create for cluster
****************************************************************************************************************/
void creatMpiClusterStruct(MPI_Datatype& clusterType)
{
	//init cluster 
	Cluster temp = {0,0,0,0,0,0};

	//Type of element in each block
	MPI_Datatype type[6] = {MPI_INT , MPI_FLOAT , MPI_FLOAT ,MPI_FLOAT ,MPI_FLOAT ,MPI_INT};
	
	//Number of elements in each block
	int typeLen[6] = {1 , 1 , 1 , 1 , 1 ,1};
	
	//Bytes displacement in memory
	MPI_Aint typeIndices[6];
	typeIndices[0] = ( (char*)&(temp.id) ) - ( (char*)&temp );
	typeIndices[1] = ( (char*)&(temp.x) ) - ( (char*)&temp );
	typeIndices[2] = ( (char*)&(temp.y) ) - ( (char*)&temp );
	typeIndices[3] = ( (char*)&(temp.sumX) ) - ( (char*)&temp );
	typeIndices[4] = ( (char*)&(temp.sumy) ) - ( (char*)&temp );
	typeIndices[5] = ( (char*)&(temp.pointsCounter) ) - ( (char*)&temp );

	//Create the type and commit so MPI now will know what the struct is.
	MPI_Type_struct(6 , typeLen, typeIndices , type , &clusterType);
	MPI_Type_commit(&clusterType);
}

/****************************************************************************************************************
Function that the Master run before and after the main algorithem calculation are made. 
It handles the file read/write,getting the input, broadcasting the data to slaves and CUDA device
- myid - id of proccessor (usually 0 because it is the master)
- numprocs - number of total proccessors/slaves
****************************************************************************************************************/
void Master(int myid , int numprocs)
{
	//predefined paths for input/output files
	char* readFilePath = "C:\\Users\\eldad\\Desktop\\liel_final_project\\K_Means_Parallel_Final\\K_Means_Parallel_Final\\giladexample.txt";
	char* writeFilePath = "C:\\Users\\eldad\\Desktop\\liel_final_project\\K_Means_Parallel_Final\\K_Means_Parallel_Final\\k_means_output.txt";
	float startTime,endTime;
	float minimumDistanceBetweenClusters,minimumDistanceTime;
	Cluster* clustersAtMinimumDistance;

	//Create the MPI data types for Point/Cluster
	MPI_Datatype typePoint,typeCluster;
	creatMpiPointStruct(typePoint);
	creatMpiClusterStruct(typeCluster);

	int numberOfPoint,numberOfClusers,numberOfIteration;
	float dt,maxT,run_time;

	//Read all the data from the input file
	Point* allPointsArray;
	allPointsArray = readFromFile( &numberOfPoint , &numberOfClusers , &dt , &maxT , &numberOfIteration , readFilePath);


	//If we have more than 1 Computer, the Master sends all the data it read from file to slaves
	if(numprocs > 1)
		brodcastParameters(&numberOfPoint , allPointsArray , &numberOfClusers , &dt ,&maxT , &numberOfIteration , myid , typePoint);

	//The master copy all the points to it's CUDA device memory
	copyPointArrayToCudaDevice(allPointsArray , numberOfPoint);

	//The master allocate memory for it's minimum distance clusters array
	clustersAtMinimumDistance = (Cluster*)malloc(numberOfClusers * sizeof(Cluster) );

	//Declare startTime for TOTAL RUNTIME output
	startTime = (float)MPI_Wtime();

	//The master calls the mainAlgorithem that runs the K-Means and CUDA usage.
	mainAlgoritem(clustersAtMinimumDistance , &minimumDistanceBetweenClusters , &minimumDistanceTime , allPointsArray , numberOfPoint , numberOfClusers , dt , maxT , myid , numprocs , numberOfIteration);

	//If we have more than 1 computer/proccessors, the Slave checks who has the minimum distance between the ending clusters and sets the minimum 
	//that we will save to file
	if(numprocs > 1)
	{
		comparingValuesWithOtherComputers(clustersAtMinimumDistance , &minimumDistanceBetweenClusters , &minimumDistanceTime , numberOfClusers , myid , numprocs ,typeCluster);
	}
	
	//The <aster calculate the TOTAL RUN TIME
	endTime = (float)MPI_Wtime();
	run_time = endTime - startTime;

	//The Master writes to file the Results of the program
	writeToFile(numberOfClusers, clustersAtMinimumDistance , minimumDistanceBetweenClusters , minimumDistanceTime , run_time , writeFilePath);

	//The master release all the Memory allocated by him and tell the cuda to release it's memory too.
	releaseCudaMemory();
	free(clustersAtMinimumDistance);
	free(allPointsArray);

	cout << "finish !" <<endl;
		
}

/****************************************************************************************************************
Function that each slave run before and after the main algorithem calculation are made.
- myid - id of slave proccessor 
- numprocs - number of total proccessors/slaves
****************************************************************************************************************/
void Slave(int myid , int numprocs)
{
	float minimumDistanceBetweenClusters,minimumDistanceTime;
	Cluster* clustersAtMinimumDistance;

	//Create the MPI data types for Point/Cluster
	MPI_Datatype typePoint,typeCluster;
	creatMpiPointStruct(typePoint);
	creatMpiClusterStruct(typeCluster);

	int numberOfPoint,numberOfClusers,numberOfIteration;
	float dt,maxT;
	Point* allPointsArray;

	//The slave recieve all the data to use from Master
	brodcastParameters(&numberOfPoint , allPointsArray , &numberOfClusers , &dt ,&maxT , &numberOfIteration , myid , typePoint);

	//The slave copy all the points to it's CUDA device memory
	copyPointArrayToCudaDevice(allPointsArray , numberOfPoint);

	//The slave allocate memory for it's minimum distance clusters array
	clustersAtMinimumDistance = (Cluster*)malloc(numberOfClusers * sizeof(Cluster) );

	//The slave calls the mainAlgorithem that runs the K-Means and CUDA usage.
	mainAlgoritem(clustersAtMinimumDistance , &minimumDistanceBetweenClusters , &minimumDistanceTime , allPointsArray , numberOfPoint , numberOfClusers , dt , maxT , myid , numprocs , numberOfIteration);

	//The slave returns the results to the Master .
	comparingValuesWithOtherComputers(clustersAtMinimumDistance , &minimumDistanceBetweenClusters , &minimumDistanceTime , numberOfClusers , myid , numprocs ,typeCluster);

	//free memory !!!!
	releaseCudaMemory();
	free(clustersAtMinimumDistance);
	free(allPointsArray);


}

/****************************************************************************************************************
Function writes to file data at the end of the program:
-numberOfClusers = number of clusters we had
-minimumDistanceClustersArray = the cluster array at minimum dt we found
-minimumDistance = the minimum distance itself between clusters during the runtime
-numberOfIteration = iterations it took to finish
-writeFileName = file name to write to
****************************************************************************************************************/
void writeToFile(int numberOfClusters , Cluster* minimumDistanceClustersArray , float minimumDistance , float minimumDistanceDT , float run_time , char* writeFileName)
{
	FILE* writeFile = fopen(writeFileName , "w");

	char* runTimeText = "Run Time :";
	fprintf(writeFile , "%s %f \n" , runTimeText , run_time);

	char* minimumDistanceText = "d= ";
	fprintf(writeFile , "%s %f \n" , minimumDistanceText , minimumDistance);

	char* minimunDtText = "t= ";
	fprintf(writeFile , "%s %f \n" , minimunDtText , minimumDistanceDT);

	char* clusterCenterText = "Centers of the Clusters :";
	fprintf(writeFile , "%s  \n" , clusterCenterText );

	for(int i = 0 ; i < numberOfClusters ; i++)
	{
		fprintf(writeFile," %f %f \n",minimumDistanceClustersArray[i].x ,minimumDistanceClustersArray[i].y);
	}

	fclose(writeFile);
}

/****************************************************************************************************************
Function reads from given files data and initializes points with relevent info:
-numberOfPoints = number of points to read
-numberOfClusers = number of cluster to read
-dt = delta time 
-maxT = maximum runtime
-numberOfIteration = iteration per DT
-readFileName = file name to read from + path(const)
****************************************************************************************************************/
Point* readFromFile(int* numberOfPoints ,int* numberOfClusers , float* dt , float* maxT , int* numberOfIteration,char* readFileName)
{
	FILE* readFile = fopen(readFileName,"r");

	fscanf(readFile ," %d %d %f %f %d" , numberOfPoints , numberOfClusers , dt ,maxT ,numberOfIteration );

	//allocate array of points
	Point* allPointsArray = (Point*)malloc( (*numberOfPoints) * sizeof(Point));

	int id;
	float a , b , r;
	for(int i = 0 ; i < *numberOfPoints ; i++)
	{
		fscanf(readFile, "%d %f %f %f" , &id , &a , &b , &r);
		//create temporary point structure with given data of x,y of actual point on circle is 0,0 since we hadnt had iterations yet
		Point tempPoint = {id,a,b,r,0,0};
		//deep copy the points structure
		allPointsArray[i] = tempPoint;
	}

	fclose(readFile);

	return allPointsArray;
}


/***************************************************************************************************************
Function return euclidian distance between clusters
-clusterOne - the first cluster
-clusterTwo - the second cluster
***************************************************************************************************************/
float distanceBetweenClusters(Cluster& clusterOne, Cluster& clusterTwo)
{
	float x = clusterOne.x - clusterTwo.x;
	float y = clusterOne.y - clusterTwo.y;
	return sqrt( x*x + y*y );
}


/***************************************************************************************************************
Function return euclidian distance between cluster and a certain point
-cluster - the cluster
-point - the point
***************************************************************************************************************/

float distanceBetweenPointAndCluster(Point& point, Cluster& cluster)
{
	float x = point.x - cluster.x;
	float y = point.y - cluster.y;
	return sqrt( x*x + y*y );
}
