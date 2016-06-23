
#include "spn.h"


typedef struct layer{
	node **nds;
	void *prev;
	void *next;
} layer;



int main(int argc,char **argv)    
{   
	FILE *fd1, *fd2;

	int n,res;
	spn spn1;
	float 		tAllocate,tBuild,tForward,tBackProp;
	cudaEvent_t bAllocate,bBuild,bForward,bBackProp;
	cudaEvent_t eAllocate,eBuild,eForward,eBackProp;


	if(argc==2){
		n=atoi(argv[1]);
		res=1;
	}else if(argc==3){
		n=atoi(argv[1]);
		res=atoi(argv[2]);
	}else{
		n=5;
		res=1;
	}

	char labels[10000];
	char imgs[n*n*10000];
	char *d_imgs;
	char *d_labels;

	int matrix_bytes=n*n*sizeof(void*);
	float 	*d_input;
	node 	**d_maps;


	fd2=fopen("/media/german/Shared/Workspace/Datasets/Dummy/dummy10x10-10000-imgs","r");
	fd1=fopen("/media/german/Shared/Workspace/Datasets/Dummy/dummy10x10-10000-labels","r");

	fread(labels,sizeof(char),10000,fd1);
	fread(imgs,sizeof(char),10000*n*n,fd2);

	fclose(fd1);
	fclose(fd2);

	error=cudaMalloc((void **) &d_imgs, 10000*n*n*sizeof(char));
	if (error != cudaSuccess){
        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
	}

	error=cudaMalloc((void **) &d_labels, 10000*sizeof(char));
	if (error != cudaSuccess){
        printf("cudaMalloc reurned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
	}

	error=cudaMemcpy(d_imgs,imgs,10000*n*n*sizeof(char),cudaMemcpyHostToDevice);
	if (error != cudaSuccess){
	    printf("cudaMemcpy returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	    exit(EXIT_FAILURE);
	}

	error=cudaMemcpy(d_labels,labels,10000*sizeof(char),cudaMemcpyHostToDevice);
	if (error != cudaSuccess){
	    printf("cudaMemcpy returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	    exit(EXIT_FAILURE);
	}



	error =cudaEventCreate(&bAllocate);
	if (error != cudaSuccess){
        printf("cudaEventCreate returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__-2);
        exit(EXIT_FAILURE);
	}

	error =cudaEventCreate(&bBuild);
	if (error != cudaSuccess){
        printf("cudaEventCreate returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__-2);
        exit(EXIT_FAILURE);
	}
	
	error =cudaEventCreate(&bForward);
	if (error != cudaSuccess){
        printf("cudaEventCreate returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__-2);
        exit(EXIT_FAILURE);
	}
	
	error =cudaEventCreate(&bBackProp);
	if (error != cudaSuccess){
        printf("cudaEventCreate returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__-2);
        exit(EXIT_FAILURE);
	}

	error =cudaEventCreate(&eAllocate);
	if (error != cudaSuccess){
        printf("cudaEventCreate returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__-2);
        exit(EXIT_FAILURE);
	}

	error =cudaEventCreate(&eBuild);
	if (error != cudaSuccess){
        printf("cudaEventCreate returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__-2);
        exit(EXIT_FAILURE);
	}
	
	error =cudaEventCreate(&eForward);
	if (error != cudaSuccess){
        printf("cudaEventCreate returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__-2);
        exit(EXIT_FAILURE);
	}
	
	error =cudaEventCreate(&eBackProp);
	if (error != cudaSuccess){
        printf("cudaEventCreate returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__-2);
        exit(EXIT_FAILURE);
	}

	
	error=cudaMalloc((void **) &d_input, n*n*sizeof(float));
	if (error != cudaSuccess){
        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
	}




	/*-----------------------------------------------------------------------------------------------------------------
	ALLOCATE
	-----------------------------------------------------------------------------------------------------------------*/

	error=cudaEventRecord(bAllocate,0);
	if (error != cudaSuccess){
	    printf("cudaMemcpy returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	    exit(EXIT_FAILURE);
	}	


	error=cudaMalloc((void **) &d_maps, matrix_bytes);
	if (error != cudaSuccess){
        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
	}

	spn_allocate(&spn1,d_maps,n,res);
	printf("TAMAÑO DE LA SPN: %ld",spn1.size);
	
	error=cudaEventRecord(eAllocate,0);
	if (error != cudaSuccess){
	    printf("cudaEventRecord returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	    exit(EXIT_FAILURE);
	}

	/*-----------------------------------------------------------------------------------------------------------------
	BUILD
	-----------------------------------------------------------------------------------------------------------------*/


	printf("-------------------------------------\n");
	printf("TOTAL=%d, (Exp=%d)\n", 0, n*(n+1)*n*(n+1)/4);
	printf("-------------------------------------\n");
	

	printf("%d\n",n );

	error=cudaEventRecord(bBuild,0);
	if (error != cudaSuccess){
	    printf("cudaEventRecord returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	    exit(EXIT_FAILURE);
	}

	//spn_build(&spn1, d_maps);
	
	error = cudaDeviceSynchronize();
	if (error != cudaSuccess){
	    printf("cudaMemcpy returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	    exit(EXIT_FAILURE);
	}

	error=cudaEventRecord(eBuild,0);
	if (error != cudaSuccess){
	    printf("cudaEventRecord returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	    exit(EXIT_FAILURE);
	}


	printf("-------------------------------------\n");
	printf("FORWARD\n");
	printf("-------------------------------------\n");

	error=cudaEventRecord(bForward,0);
	if (error != cudaSuccess){
	    printf("cudaEventRecord returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	    exit(EXIT_FAILURE);
	}
	for(int i=1;i<2*(n/res)+2*res-1;i++){
		updateVal<<<1,(spn1.l_count)[i]>>>((node***)((spn1.tree)+i));
	}
	error = cudaDeviceSynchronize();	
	if (error != cudaSuccess){
		    printf("Kernel updateVal returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		    exit(EXIT_FAILURE);
	}
		
	


	error=cudaEventRecord(eForward,0);
	if (error != cudaSuccess){
	    printf("cudaEventRecord returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	    exit(EXIT_FAILURE);
	}

	printf("-------------------------------------\n");
	printf("BackProp\n");
	printf("-------------------------------------\n");

	error=cudaEventRecord(bBackProp,0);
	if (error != cudaSuccess){
	    printf("cudaEventRecord returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	    exit(EXIT_FAILURE);
	}


	for(int i=2*(n/res)+2*res-4;i>0;i--){
		backProp<<<1,(spn1.l_count)[i]>>>((node***)((spn1.tree)+i));
	
	}
	error = cudaDeviceSynchronize();
	if (error != cudaSuccess){
	    printf("cudaMemcpy returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	    exit(EXIT_FAILURE);
	}
	
	error=cudaEventRecord(eBackProp,0);
	if (error != cudaSuccess){
	    printf("cudaEventRecord returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	    exit(EXIT_FAILURE);
	}




	error =cudaEventSynchronize(eAllocate);
	if (error != cudaSuccess){
        printf("cudaEventSynchronize returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__-2);
        exit(EXIT_FAILURE);
	}

	error =cudaEventSynchronize(eBuild);
	if (error != cudaSuccess){
        printf("cudaEventSynchronize returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__-2);
        exit(EXIT_FAILURE);
	}
	
	error =cudaEventSynchronize(eForward);
	if (error != cudaSuccess){
        printf("cudaEventSynchronize returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__-2);
        exit(EXIT_FAILURE);
	}

	error =cudaEventSynchronize(eBackProp);
	if (error != cudaSuccess){
        printf("cudaEventSynchronize returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__-2);
        exit(EXIT_FAILURE);
	}


	error =cudaEventElapsedTime(&tAllocate,bAllocate,eAllocate);
	if (error != cudaSuccess){
        printf("cudaEventSynchronize returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__-2);
        exit(EXIT_FAILURE);
	}

	error =cudaEventElapsedTime(&tBuild,bBuild,eBuild);
	if (error != cudaSuccess){
        printf("cudaEventSynchronize returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__-2);
        exit(EXIT_FAILURE);
	}

	error =cudaEventElapsedTime(&tForward,bForward,eForward);
	if (error != cudaSuccess){
        printf("cudaEventSynchronize returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__-2);
        exit(EXIT_FAILURE);
	}

	error =cudaEventElapsedTime(&tBackProp,bBackProp,eBackProp);
	if (error != cudaSuccess){
        printf("cudaEventSynchronize returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__-2);
        exit(EXIT_FAILURE);
	}
	printf("-------------------------------------\n");
	printf("Programa Finalizado\n");
	printf("-------------------------------------\n");
	printf("Tiempos:\n");
	printf("	Allocation:\t%f\n",tAllocate);
	printf("	Build:\t\t%f\n",tBuild);
	printf("	Forward:\t%f\n",tForward);
	printf("	BackProp:\t%f\n",tBackProp);
	printf("	Total:\t%f\n",tBackProp+tForward);


}

