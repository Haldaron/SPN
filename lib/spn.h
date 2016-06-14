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
	int dim1;
	int *d_dim1;
	int *d_l_count;
	int *l_count;
	int *d_res;
} spn;

cudaError_t error;

__global__ void createNode( int *d_n, node **map, node ***tree ,int *l_count){
	
	int pos;
	int n=*d_n;
	int i=n-blockDim.x;
	int j=n-blockDim.y;
	int x=threadIdx.x;
	int y=threadIdx.y;
	int t=i+j;//Number of product nodes
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


__global__ void createLeave( int *d_n, int *d_res, int *d_x, int *d_y, node **map, node ***tree ,int *l_count){
	
	int n=*d_n;
	int res=*d_res;
	int i=res-blockDim.x;
	int j=res-blockDim.y;
	int x=*d_x;
	int y=*d_y;
	int x2=threadIdx.x;
	int y2=threadIdx.y;
	
	node *nd;

	nd=&map[j*res+i][y2*blockDim.x+x2];


	(*nd).h=1;
	(*nd).w=1;
	(*nd).x=x2;
	(*nd).y=y2;	

	(*nd).val=1.0f;

	if(j==n && i==n){
		(*nd).dv=1.0f;
	}else{
		(*nd).dv=0.0f;
	}	


	// atomicAdd(&l_count[i+j],1);

	tree[i+j][(y*res+y2)*n+(x*res+x2)]=nd;
}




__global__ void input(node ***p_layer, char *input){
	int i=threadIdx.x;
	node **layer=*p_layer;
	node *nd=layer[i];
	
	(*nd).val=1.0f;
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
	printf("%d\n",val );
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

int spn_create_map(int n, node **d_maps, int *l_count){
	int i,j,i2,j2,r,c;
	for(i=0; i<n;i++){
		for(j=0; j<n;j++){

			r=n-i;
			c=n-j;
			node *mat;

			error=cudaMalloc(&mat,r*c*sizeof(node));
			if (error != cudaSuccess){
		        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		        exit(EXIT_FAILURE);
		    }

		    error=cudaMemcpy(&d_maps[i*n+j],&mat,sizeof(void*), cudaMemcpyHostToDevice);
			if (error != cudaSuccess){
		        printf("cudaMemcpy returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		        exit(EXIT_FAILURE);
		    }
		    l_count[i+j]+=r*c;

		    for(i2=0;i2<r;i2++){
				for(j2=0;j2<c;j2++){


					float *vals;
					node **chs;
					error=cudaMalloc(&vals,4*(i+j)*sizeof(float));
					if (error != cudaSuccess){
				        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
				        exit(EXIT_FAILURE);
				    }

				    error=cudaMalloc(&chs,2*(i+j)*sizeof(float));
					if (error != cudaSuccess){
				        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
				        exit(EXIT_FAILURE);
				    }

				    error=cudaMemcpy(&(mat[j2*r+i2].chs),&chs,sizeof(void*), cudaMemcpyHostToDevice);
					if (error != cudaSuccess){
				        printf("cudaMemcpy returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
				        exit(EXIT_FAILURE);
				    }
					
				    error=cudaMemcpy(&(mat[j2*r+i2].ws),&vals,sizeof(void*), cudaMemcpyHostToDevice);
					if (error != cudaSuccess){
				        printf("cudaMemcpy returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
				        exit(EXIT_FAILURE);
				    }
				}			
			}
		}
	}
	return 0;
}

int spn_allocate(spn  *ptr, node **d_maps, int d, int res){
	int 	n=d/res;
	node 	***smaps=(node***)malloc(n*n*sizeof(void*));
	int 	*l_count= (int *)malloc((2*n+2*res-1)*sizeof(int));
	int 	i;

	printf("n: %d, res: %d\n",n,res );
	(*ptr).dim1=d;
	(*ptr).l_count=l_count;

	error=cudaMalloc((void **) &((*ptr).d_dim1), sizeof(int));
	if (error != cudaSuccess){
        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
	}
	
	error=cudaMalloc((void **) &((*ptr).d_res), sizeof(int));
	if (error != cudaSuccess){
        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
	}

	error=cudaMalloc((void **) &((*ptr).tree), (2*n+2*res-1)*sizeof(void*));
	if (error != cudaSuccess){
        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
	}

	error=cudaMalloc((void **) &((*ptr).d_l_count), (2*n+2*res-1)*sizeof(void*));
	if (error != cudaSuccess){
        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
	}

	error=cudaMemset((*ptr).d_l_count,0,(2*n+2*res-1)*sizeof(void*));
	if (error != cudaSuccess){
	    printf("cudaMemset returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	    exit(EXIT_FAILURE);
	}

	error=cudaMemcpy((*ptr).d_dim1,&n,sizeof(int),cudaMemcpyHostToDevice);
	if (error != cudaSuccess){
	    printf("cudaMemcpy returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	    exit(EXIT_FAILURE);
	}

	error=cudaMemcpy((*ptr).d_res,&res,sizeof(int),cudaMemcpyHostToDevice);
	if (error != cudaSuccess){
	    printf("cudaMemcpy returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	    exit(EXIT_FAILURE);
	}

	memset(l_count,0,(2*n+2*res-1)*sizeof(int));

	spn_create_map(n,d_maps,l_count+2*res-1);
	printf("TodoBien\n");
	sleep(1);

	error = cudaDeviceSynchronize();
	if (error != cudaSuccess){
	    printf("cudaDeviceSynchronize returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	    exit(EXIT_FAILURE);
	}

	
	printf("building tree\n");

	for(i=2*res-1;i<2*n+2*res-1;i++){
		node **layer;
		error=cudaMalloc(&layer,l_count[i]*sizeof(void*));
		if (error != cudaSuccess){
	        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	        exit(EXIT_FAILURE);
	    }

		
	    error=cudaMemcpy(&((*ptr).tree[i]),&layer,sizeof(void*), cudaMemcpyHostToDevice);
		if (error != cudaSuccess){
	        printf("cudaMemcpy returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	        exit(EXIT_FAILURE);
	    }
	    printf("Layer:%d, %d\n",i,l_count[i]);
	}	


	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++){
			int r=n-i;
			int c=n-j;
			dim3 THREAD_DIM (r,c);
			createNode<<<1,THREAD_DIM>>>((*ptr).d_dim1,(node**)d_maps, (*ptr).tree+2*res-1, (*ptr).d_l_count+2*res-1);
		}
	}

	printf("TodoBien\n");
	sleep(1);

	error = cudaDeviceSynchronize();
	if (error != cudaSuccess){
	    printf("cudaDeviceSynchronize returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	    exit(EXIT_FAILURE);
	}


			printf("poblando submap\n");

	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++){
			node **smap=(node**)malloc(sizeof(void*));
			error=cudaMalloc(&smap,res*res*sizeof(void*));
			if (error != cudaSuccess){
		        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        	    exit(EXIT_FAILURE);

		    }
		    

		    spn_create_map(res,smap,l_count);

		    smaps[i*n+j]=smap;

		    error = cudaDeviceSynchronize();
			if (error != cudaSuccess){
			    printf("cudaDeviceSynchronize returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
			    exit(EXIT_FAILURE);
			}
		}
	}


	for(i=0;i<2*res-1;i++){
		node **layer;
		error=cudaMalloc(&layer,l_count[i]*sizeof(void*));
		if (error != cudaSuccess){
	        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	        exit(EXIT_FAILURE);
	    }

		
	    error=cudaMemcpy(&((*ptr).tree[i]),&layer,sizeof(void*), cudaMemcpyHostToDevice);
		if (error != cudaSuccess){
	        printf("cudaMemcpy returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	        exit(EXIT_FAILURE);
	    }
	    printf("Layer:%d, %d\n",i,l_count[i]);
	}	




	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++){
			node ** smap=smaps[i*n+j];
			int *d_x,*d_y;


		    error=cudaMalloc(&d_x,sizeof(int));
			if (error != cudaSuccess){
		        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		        exit(EXIT_FAILURE);
		    }

		    error=cudaMalloc(&d_y,sizeof(int));
			if (error != cudaSuccess){
		        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		        exit(EXIT_FAILURE);
		    }		

		    error=cudaMemcpy(d_y,&i,sizeof(int), cudaMemcpyHostToDevice);
			if (error != cudaSuccess){
		        printf("cudaMemcpy returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		        exit(EXIT_FAILURE);
		    }

		    error=cudaMemcpy(d_x,&j,sizeof(int), cudaMemcpyHostToDevice);
			if (error != cudaSuccess){
		        printf("cudaMemcpy returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		        exit(EXIT_FAILURE);
		    }

			for(int i=0;i<res;i++){
				for(int j=0;j<res;j++){

					int r=res-i;
					int c=res-j;
					dim3 THREAD_DIM (r,c);
					if(i!=0&&j!=0){

						createNode<<<1,THREAD_DIM>>>((*ptr).d_res,(node**)smap, (*ptr).tree, (*ptr).d_l_count);
					}else{
						createLeave<<<1,THREAD_DIM>>>((*ptr).d_dim1,(*ptr).d_res,d_x,d_y,(node**)smap, (*ptr).tree, (*ptr).d_l_count);
					}
				}
			}

			error = cudaDeviceSynchronize();
			if (error != cudaSuccess){
			    printf("cudaDeviceSynchronize returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
			    exit(EXIT_FAILURE);
			}
		}
	}



	return 0;
}





#endif 