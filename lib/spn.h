#ifndef	SPN_H
#define SPN_H

#include <stdio.h>
#include <unistd.h>
#include <curand_kernel.h>
#include <curand.h>
#include <math.h>


/*------------------------------------------------------------------------------------



-------------------------------------------------------------------------------------*/
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
	node ***d_tree;
	long size;
	int h_dim1;
	int *d_dim1;
	int *d_l_count;
	int *h_l_count;
	int *d_res;
} spn;

cudaError_t error;

__global__ void spn_createNode( int *d_n, node **map, node ***tree ,int *l_count){
	
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


__global__ void spn_createLeave( int *d_n, int *d_res, int *d_x, int *d_y, node **map, node ***tree ,int *l_count){
	
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




__global__ void spn_input(node ***p_layer, float *input){
	int i=threadIdx.x;
	node **layer=*p_layer;
	node *nd=layer[i];
	
	(*nd).val=exp(-pow(input[i],2));
	return;
}


__global__ void spn_updateVal(node ***p_layer){
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


__global__ void spn_updateWeights(node ***p_layer,float *eta){
	int t;
	int i=threadIdx.x;
	float l_eta=*eta;
	node **layer=*p_layer;
	node *nd=layer[i];
	node **chs= (node**)(*nd).chs;
	float temp=0;
	t=(*nd).w+(*nd).h-2;


	for(int k=0;k<t;k++){
		(*nd).ws[k]+=l_eta*(*nd).dw[k];
		temp+=(*nd).ws[k];
	}
	
	for(int k=0;k<t;k++){
		(*nd).ws[k]=(*nd).ws[k]/temp;
	}

}

__global__ void spn_delta(node ***p_layer,float *pred,int *label){
	node **layer=*p_layer;
	node *nd=layer[0];

	(*nd).dv=pow(*label-*pred,2);

}


__global__ void spn_updateDer(node ***p_layer){
	int t;
	int i=threadIdx.x;
	node **layer=*p_layer;
	node *nd=layer[0];
	node **chs= (node**)(*nd).chs;
	t=(*nd).w+(*nd).h-2;

	//Update Product Node derivatives
	for(int k=0;k<t;k++){
		(*nd).dp[k]+=(*nd).val*(*nd).ws[k];
		(*nd).dw[k]=(*nd).val*(*nd).ps[k];
	}

	for(int k=0;k<t;k++){
		(*chs[2*k]).dv+=(*nd).dp[k]*(*chs[2*k+1]).val;
		(*chs[2*k+1]).dv+=(*nd).dp[k]*(*chs[2*k]).val;
	}

}

long spn_createMap(int n, node **d_maps, int *l_count){
	int i,j,i2,j2,r,c;
	long size=0;
	for(i=0; i<n;i++){
		for(j=0; j<n;j++){

			r=n-i;
			c=n-j;
			node *mat;
			
			size+=r*c*sizeof(node);
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
					size+=4*(i+j)*sizeof(float);
					error=cudaMalloc(&vals,4*(i+j)*sizeof(float));
					if (error != cudaSuccess){
				        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
				        exit(EXIT_FAILURE);
				    }

					size+=2*(i+j)*sizeof(float);
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
	return size;
}

int spn_buildPDSPN(spn  *ptr, int d, int res){
	int 	n=d/res;
	node 	**d_maps;
	node 	***smaps=(node***)malloc(n*n*sizeof(void*));
	int 	*l_count= (int *)malloc((2*n+2*res-1)*sizeof(int));
	long	size=0;

	if(d<=re){
		printf("Error en spn_buildPDSPN: la resolución debe ser menor que la dimensión de los datos de entrada");
		return -1;
	}
	
	(*ptr).h_dim1=d;
	(*ptr).h_l_count=l_count;

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

	size+=(2*n+2*res-1)*sizeof(void*);
	error=cudaMalloc((void **) &((*ptr).tree), (2*n+2*res-1)*sizeof(void*));
	if (error != cudaSuccess){
        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
	}
	
	error=cudaMalloc((void **) &((*ptr).d_maps), n*n*sizeof(void*));
	if (error != cudaSuccess){
        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
	}

	size+=(2*n+2*res-1)*sizeof(void*);
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

	size+=spn_create_map(n,d_maps,l_count+2*res-1);


	error = cudaDeviceSynchronize();
	if (error != cudaSuccess){
	    printf("cudaDeviceSynchronize returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	    exit(EXIT_FAILURE);
	}

	
	printf("building tree...\n");

	for(int i=2*res-1;i<2*n+2*res-1;i++){
		node **layer;
		size+=l_count[i]*sizeof(void*);
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


	error = cudaDeviceSynchronize();
	if (error != cudaSuccess){
	    printf("cudaDeviceSynchronize returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	    exit(EXIT_FAILURE);
	}


	printf("poblando submap\n");

	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++){
			node **smap=(node**)malloc(sizeof(void*));
			size+=res*res*sizeof(void*);
			error=cudaMalloc(&smap,res*res*sizeof(void*));
			if (error != cudaSuccess){
		        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        	    exit(EXIT_FAILURE);

		    }
		    

		    size+=spn_create_map(res,smap,l_count);

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
		size+=l_count[i]*sizeof(void*);
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


	(*ptr).size=size;

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

			for(int k=0;k<res;k++){
				for(int l=0;l<res;l++){

					int r=res-k;
					int c=res-l;
					dim3 THREAD_DIM (r,c);
					if(k!=0&&l!=0){

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

void spn_forward(spn *ptr, int n, float *data, float *labels){
	int d=(*ptr).h_dim1;
	int res=(*ptr).h_res;
	for(int j=0;j<n;j++){
		spn_input<<<1,(((*ptr).h_l_count)[0])>>>((*ptr).d_tree,data+j*d*d);
		for(int i=1;i<2*(d/res)+2*res-3;i++){
			spn_updateVal<<<1,((*ptr).h_l_count)[i]>>>((node***)(((*ptr).d_tree)+i));
		}
	}
}

void spn_backProp(spn *ptr, float *pred, float *label){
	int d=(*ptr).h_dim1;
	int res=(*ptr).h_res;
	
	spn_delta<<<1,(((*ptr).h_l_count)[2*(d/res)+2*res-3])>>>((*ptr).d_tree)+2*(d/res)+2*res-3,pred,label);
	for(int i=2*(d/res)+2*res-3;i>0;i--){
		spn_updateDer<<<1,((*ptr).h_l_count)[i]>>>((node***)(((*ptr).d_tree)+i));
	}
}

void spn_updateSPNWeights(spn *ptr, *float eta){
	int d=(*ptr).h_dim1;
	int res=(*ptr).h_res;
	
	for(int i=2*(d/res)+2*res-3;i>0;i--){
		spn_updateWeights<<<1,((*ptr).h_l_count)[i]>>>((node***)(((*ptr).d_tree)+i),eta);
	}
}

void spn_trainOGD(spn *ptr, int n,float *data, float *labels, float eta, float eor, int maxepoch){
	float *pred;
	int d=(*ptr).h_dim;
	int n_er=0;
	float d_eta;
	
	error=cudaMalloc((void **) &pred, n*sizeof(float));
	if (error != cudaSuccess){
        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
	}
	
	error=cudaMalloc((void **) &e_eta, sizeof(float));
	if (error != cudaSuccess){
        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
	}
	
	error=cudaMemcpy(d_eta,eta,sizeof(int),cudaMemcpyHostToDevice);
	if (error != cudaSuccess){
	    printf("cudaMemcpy returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	    exit(EXIT_FAILURE);
	}
	for(int j=0;j<maxepoch;j++){
		for(int i=0; i<n; i++){
			spn_forward(ptr,1,data+i*d*d,pred+i);
			spn_backProp(ptr,pred+i,labels+i);
			spn_updateSPNWeights(ptr, d_eta);
		}
	}
}

#endif 
