#ifndef	SPN_H
#define SPN_H

#include <stdio.h>
#include <unistd.h>
#include <curand_kernel.h>
#include <curand.h>
#include <math.h>

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

typedef struct spn{
	node ***tree;
	int *dim1;
	int *l_count;
} spn;

cudaError_t error;

__global__ void createNode( int *d_n, node **map, node ***tree ,int *l_count){
	
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

	(*nd).val=1.0f;

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




__global__ void delta(node ***p_layer, char *label){
	node **layer=*p_layer;
	node *nd=layer[0];

	(*nd).dv=*label-(*nd).dv;

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



int spn_allocate(spn  *ptr, int n){
	node 	**d_maps;
	node 	***d_tree;
	int 	l_count[2*n-1];
	int i,j,i2,j2,r,c;
	int matrix_bytes=n*n*sizeof(void*);

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
	return 0;
}


int spn_build(spn *ptr){
	int *n=(*ptr).dim1;
	node 	**d_maps;
	node 	***d_tree;

	for(int i=0;i<*n;i++){
		for(int j=0;j<*n;j++){
			int r=*n-i;
			int c=*n-j;
			dim3 THREAD_DIM (r,c);
			createNode<<<1,THREAD_DIM>>>(n,(node**)d_maps, d_tree, (*ptr).l_count);
		}
	}

	error = cudaDeviceSynchronize();
	if (error != cudaSuccess){
	    printf("cudaMemcpy returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	    exit(EXIT_FAILURE);
	}
	return 0;

}
#endif 