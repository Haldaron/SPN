#include <stdio.h>
#include <unistd.h>
#include <curand_kernel.h>
#include <curand.h>
#include <math.h>
#include <iostream>

typedef struct  node{
	float	val;
	float	dv;
	float	*ws;
	float	*ps;
	float	*dw;
	float	*dp;
	void 	**chs;
	int		x;
	int		y;
	int		w;
	int		h;
} node;


typedef struct layer{
	node **nds;
	void *prev;
	void *next;
} layer;

__global__ void createNode(  int *d_n,node **map, float *input, node ***tree ,int *l_count){
	
	int pos;
	int n=*d_n;
	int i=n-blockDim.x;
	int j=n-blockDim.y;
	int x=threadIdx.x;
	int y=threadIdx.y;
	int t=i+j;
	node *nd;

	nd=&map[j*n+i][y*blockDim.x+x];


	(*nd).h=i+1;
	(*nd).w=j+1;
	(*nd).x=x;
	(*nd).y=y;	

	if(j!=0||i!=0){
		(*nd).val=0;
	}else{
		(*nd).val=1.0f;
	}

	if(j==n && i==n){
		(*nd).dv=1.0f;
	}else{
		(*nd).dv=0.0f;
	}	

	
	(*nd).ps=(*nd).ws+t;
	(*nd).dw=(*nd).ps+t;
	(*nd).dp=(*nd).dw+t;

	for(int k=0;k<t;k++){
		(*nd).ws[k]=1.0f/(t);
	}

	//Vertical split
	for(int k=1;k<j+1;k++){

		(*nd).chs[2*k-2]=&map[(k-1)*n +i][y*(n-i)+x];
		(*nd).chs[2*k-1]=&map[(j-k)*n + i][(y+k)*(n-i)+x];
		(*nd).ps[k-1]=0;
		(*nd).dp[k-1]=0;
		(*nd).dw[k-1]=0;

	}

	//Horizontal split
	for(int k=1;k<i+1;k++){
		(*nd).chs[2*j+2*k-2]=&map[j*n+k-1][y*(n-k+1)+x];
		(*nd).chs[2*j+2*k-1]=&map[j*n+(i-k)][y*(n-i+k)+x+k];
		(*nd).ps[j+k-1]=0;
		(*nd).dp[j+k-1]=0;
		(*nd).dw[j+k-1]=0;

	}

	pos = atomicAdd(&l_count[i+j],1);

	tree[i+j][pos]=nd;

}




__global__ void input(node ***p_layer, char *input){
	int i=threadIdx.x;
	node **layer=*p_layer;
	node *nd=layer[i];
	
	(*nd).val=input[i];
	return;
}


__global__ void updateVal(node ***p_layer){
	int t;
	int i=threadIdx.x;
	float temp;
	node **layer=*p_layer;
	node *nd=layer[i];
	node **chs= (node**)(*nd).chs;
	t=(*nd).w+(*nd).h-2;
	
	float val=0;
	for(int k=0;k<t;k++){
		temp=(*nd).ws[k]*((*chs[2*k]).val)*((*chs[2*k+1]).val);
		(*nd).ps[k]=temp;
		val+=temp;
	}

	(*nd).val=val;
}




__global__ void backProp(node ***p_layer){
	int t;
	int i=threadIdx.x;
	node **layer=*p_layer;
	node *nd=layer[i];
	node **chs= (node**)(*nd).chs;
	t=(*nd).w+(*nd).h-2;

	//Update Product Node derivatives
	for(int k=0;k<t;k++){
		(*nd).dp[k]+=(*nd).dv*(*nd).ws[k];
		(*nd).dw[k]=(*nd).dv*(*nd).ps[k];
	}

	for(int k=0;k<t;k++){
		(*chs[2*k]).dv+=(*nd).dp[k]*(*chs[2*k+1]).val;
		(*chs[2*k+1]).dv+=(*nd).dp[k]*(*chs[2*k]).val;
	}

}


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

    cudaError_t error;
	float 	*d_input;
	int 	l_count[2*n-1];
	int 	*d_l_count;
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
        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
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

	for(int i=2*n-1;i>0;i--){
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

