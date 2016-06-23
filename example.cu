
#include "lib/spn.h"


int main(int argc,char **argv)    
{   


	/*-----------------------------------------------------------------------------------------------------------------
	PREPARANDO AMBIENTE DE ENTRENAMIENTO
	-----------------------------------------------------------------------------------------------------------------*/
	FILE *fd1, *fd2;

	spn spn1;						//Estructura donde se creará la SPN
	
	//Declaracion de entidades para medicion de tiempo de procesamiento
	float 		tTrain;				//
	cudaEvent_t bTrain;				//Evento CUDA para inicio de entrenamiento
	cudaEvent_t eTrain;				//Evento CUDA para finalización de entrenamiento

	int n=10; 						//El número de datos de entrada a la SPN sera nxn
	int res=2;						//Nivel de resolucion de la SPN

	float labels[10000];				//Etiquetas en Host
	float imgs[n*n*10000];			//Datos de entrada en Host
	float *d_imgs;					//Etiquetas en GPU
	float *d_labels;					//Datos de entrada en GPU


	//Cargando datos de entrenamiento
	fd2=fopen("./dummy10x10-10000-imgs","r");
	fd1=fopen("./dummy10x10-10000-labels","r");

	fread(labels,sizeof(char),10000,fd1);
	fread(imgs,sizeof(char),10000*n*n,fd2);

	fclose(fd1);
	fclose(fd2);

		
	//Reservando espacios de memoria para etiquetas y datos en la GPU
	error=cudaMalloc((void **) &d_imgs, 10000*n*n*sizeof(float));
	if (error != cudaSuccess){
        printf("cudaMalloc returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
	}

	error=cudaMalloc((void **) &d_labels, 10000*sizeof(float));
	if (error != cudaSuccess){
        printf("cudaMalloc reurned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
	}

	//Copiando etiquetas y datos a la GPU
	error=cudaMemcpy(d_imgs,imgs,10000*n*n*sizeof(float),cudaMemcpyHostToDevice);
	if (error != cudaSuccess){
	    printf("cudaMemcpy returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	    exit(EXIT_FAILURE);
	}

	error=cudaMemcpy(d_labels,labels,10000*sizeof(float),cudaMemcpyHostToDevice);
	if (error != cudaSuccess){
	    printf("cudaMemcpy returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	    exit(EXIT_FAILURE);
	}



	
	error =cudaEventCreate(&bTrain);
	if (error != cudaSuccess){
        printf("cudaEventCreate returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__-2);
        exit(EXIT_FAILURE);
	}
	
	error =cudaEventCreate(&eTrain);
	if (error != cudaSuccess){
        printf("cudaEventCreate returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__-2);
        exit(EXIT_FAILURE);
	}


	/*-----------------------------------------------------------------------------------------------------------------
	BUILD SPN
	-----------------------------------------------------------------------------------------------------------------*/


	spn_buildPDSPN(&spn1,n,res);
	printf("TAMAÑO DE LA SPN: %ld",spn_size);


	/*-----------------------------------------------------------------------------------------------------------------
	TRAIN
	-----------------------------------------------------------------------------------------------------------------*/
	

	error=cudaEventRecord(bTrain,0);//Guarda el evento de inicio de proceso de entrenamiento
	if (error != cudaSuccess){
	    printf("cudaEventRecord returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	    exit(EXIT_FAILURE);
	}

	/*Se inicia un entrenamiento de la SPN con las siguientes características:
			- 10000 datos para entrenamiento.
			- eta de 10^(-3). 
			- Error objetivo de 1%.
			- Maximo 20 recorridos al set de datos completo.
	
	*/
	
	spn_trainOGD(&spn1, 10000, d_imgs, d_labels, 0.001, 1, 20);
	
	error = cudaDeviceSynchronize(); //Se asegura de que todos los kernels que se estén ejecutando terminen antes de que la CPU continue
	if (error != cudaSuccess){
	    printf("cudaMemcpy returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	    exit(EXIT_FAILURE);
	}

	error=cudaEventRecord(eTrain,0);//Guarda el evento de finalizacion de proceso de entrenamiento
	if (error != cudaSuccess){
	    printf("cudaEventRecord returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	    exit(EXIT_FAILURE);
	}





	error =cudaEventSynchronize(bTrain);//Se asegura de que cudaEventRecord haya finalizado
	if (error != cudaSuccess){
        printf("cudaEventSynchronize returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__-2);
        exit(EXIT_FAILURE);
	}

	error =cudaEventSynchronize(eTrain);//Se asegura de que cudaEventRecord haya finalizado
	if (error != cudaSuccess){
        printf("cudaEventSynchronize returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__-2);
        exit(EXIT_FAILURE);
	}

	
	error =cudaEventElapsedTime(&tTrain,bTrain,eTrain);  //Calcula tiempo transcurrido entre eventos
	if (error != cudaSuccess){
        printf("cudaEventSynchronize returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__-2);
        exit(EXIT_FAILURE);
	}

	printf("-------------------------------------\n");
	printf("Programa Finalizado\n");
	printf("-------------------------------------\n");
	printf("	Tiempo de entrenamiento:\t%f ms\n",tTrain); //Reporta tiempo de entrenamiento


}

