#include <stdio.h>
#include <unistd.h>
#include <curand_kernel.h>
#include <curand.h>
#include <math.h>
#include <iostream>
#include "spn.h"


typedef struct layer{
	node **nds;
	void *prev;
	void *next;
} layer;



int main(int argc,char **argv)    
{   
	FILE *fd1, *fd2;

	int n,i,j,i2,j2,r,c,tot;
	int *d_n;
	float 		tAllocate,tBuild,tForward,tBackProp;
	cudaEvent_t bAllocate,bBuild,bForward,bBackProp;
	cudaEvent_t eAllocate,eBuild,eForward,eBackProp;


	if(argc==2){
		n=atoi(argv[1]);
	}else{
		n=5;
	}

	char labels[10000];
	char imgs[n*n*10000];
	char *d_imgs;
	char *d_labels;

	int matrix_bytes=n*n*sizeof(void*);

	float 	*d_input;

	node 	**d_maps;
	node 	***d_tree;


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





	memset(l_count,0,(2*n-1)*sizeof(int));
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





	error=cudaMalloc((void **) &d_tree, 2*n*sizeof(void*));
	if (error != cudaSuccess){
        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
	}

	error=cudaMalloc((void **) &d_maps, matrix_bytes);
	if (error != cudaSuccess){
        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
	}

	error=cudaMalloc((void **) &d_input, n*n*sizeof(float));
	if (error != cudaSuccess){
        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
	}

	error=cudaMalloc((void **) &d_n, sizeof(int));
	if (error != cudaSuccess){
	    printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	    exit(EXIT_FAILURE);
	}

	error=cudaMalloc((void **) &d_l_count, 2*(n)*sizeof(int));
	if (error != cudaSuccess){
	    printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	    exit(EXIT_FAILURE);
	}

	error=cudaMemset( d_l_count, 0,2*(n)*sizeof(int));
	if (error != cudaSuccess){
	    printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	    exit(EXIT_FAILURE);
	}

	error=cudaMemcpy(d_n,&n,sizeof(int),cudaMemcpyHostToDevice);
	if (error != cudaSuccess){
	    printf("cudaMemcpy returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	    exit(EXIT_FAILURE);
	}



	error=cudaEventRecord(bAllocate,0);
	if (error != cudaSuccess){
	    printf("cudaMemcpy returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	    exit(EXIT_FAILURE);
	}	

	tot=0;
	for(i=0; i<n;i++){
		for(j=0; j<n;j++){

			r=n-i;
			c=n-j;
			node *mat;

			error=cudaMalloc(&mat,r*c*sizeof(node));
			if (error != cudaSuccess){
		        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		    }

		    error=cudaMemcpy(&d_maps[i*n+j],&mat,sizeof(void*), cudaMemcpyHostToDevice);
			if (error != cudaSuccess){
		        printf("cudaMemcpy returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		    }
		    l_count[i+j]+=r*c;

		    for(i2=0;i2<r;i2++){
				for(j2=0;j2<c;j2++){


					float *vals;
					node **chs;
					error=cudaMalloc(&vals,4*(i+j)*sizeof(float));
					if (error != cudaSuccess){
				        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
				    }

				    error=cudaMalloc(&chs,2*(i+j)*sizeof(float));
					if (error != cudaSuccess){
				        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
				    }

				    error=cudaMemcpy(&(mat[j2*r+i2].chs),&chs,sizeof(void*), cudaMemcpyHostToDevice);
					if (error != cudaSuccess){
				        printf("cudaMemcpy returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
				    }
					
				    error=cudaMemcpy(&(mat[j2*r+i2].ws),&vals,sizeof(void*), cudaMemcpyHostToDevice);
					if (error != cudaSuccess){
				        printf("cudaMemcpy returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
				    }
				}			
			}
		}
	}

	printf("building tree\n");

	for(i=0;i<2*n-1;i++){
		node **layer;
		error=cudaMalloc(&layer,l_count[i]*sizeof(void*));
		if (error != cudaSuccess){
	        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	    }

		
	    error=cudaMemcpy(&(d_tree[i]),&layer,sizeof(void*), cudaMemcpyHostToDevice);
		if (error != cudaSuccess){
	        printf("cudaMemcpy returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	    }
	    printf("Layer:%d, %d\n",i,l_count[i]);
	}

	
	error=cudaEventRecord(eAllocate,0);
	if (error != cudaSuccess){
	    printf("cudaEventRecord returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	    exit(EXIT_FAILURE);
	}

	printf("-------------------------------------\n");
	printf("TOTAL=%d, (Exp=%d)\n", tot, n*(n+1)*n*(n+1)/4);
	printf("-------------------------------------\n");
	

	printf("%d\n",n );

	error=cudaEventRecord(bBuild,0);
	if (error != cudaSuccess){
	    printf("cudaEventRecord returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	    exit(EXIT_FAILURE);
	}

	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++){
			int r=n-i;
			int c=n-j;
			dim3 THREAD_DIM (r,c);
			createNode<<<1,THREAD_DIM>>>(d_n,(node**)d_maps, d_input, d_tree, d_l_count);
		}
	}
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

	input<<<1,l_count[i]>>>((node***)(d_tree+i),d_imgs);

	for(int i=1;i<2*n-1;i++){
		printf("layer: %i, count: %d\n", i, l_count[i]);
		updateVal<<<1,l_count[i]>>>((node***)(d_tree+i));
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

	delta<<<1,1>>>((node***)(d_tree+2*n-2),d_labels);

	for(int i=2*n-3;i>0;i--){
		printf("layer: %i, count: %d\n", i, l_count[i]);
		backProp<<<1,l_count[i]>>>((node***)(d_tree+i));
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


}

