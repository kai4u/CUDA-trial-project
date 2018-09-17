#include <iostream>
#include <algorithm>
#include <random>

#define HANDLE_ERROR( err ) ( HandleError( err, __FILE__, __LINE__ ) )

static void HandleError( cudaError_t err, const char *file, int line )
{
    if (err != cudaSuccess)
      {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

__global__ void elementwiseAnd(int* a, int* b, int* res){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	res[index] = a[index] && b[index];
}

void initMatrix(int* a, size_t n_row, size_t n_col){	

    std::default_random_engine gen;
	std::bernoulli_distribution dist(0.5);    


	for(size_t i = 0; i < n_row; i++)
		for(size_t j = 0; j < n_col; j++){
			size_t index = j + i * n_col;
			a[index] = (dist(gen));
		}
}

int main(){
	
	using namespace std;
	
	size_t n_row, n_col;
	cin >> n_row >> n_col;
    size_t size = n_row * n_col * sizeof(int);
	
	int *c, *res;
    a = (bool *) malloc(size);initMatrix(a, n_row, n_col);
    b = (bool *) malloc(size);initMatrix(b, n_row, n_col);
    c = (int *) malloc(size);
    res = (int *) malloc(size);
    
	for(size_t i = 0; i < n_row; i++)
		for(size_t j = 0; j < n_col; j++){
			size_t index = j + i * n_col;
			c[index] = a[index] && b[index];
		}
    
   
    cout << endl;
    
	int *d_a, *d_b, *d_c;	

	cudaMalloc((void **) &d_a, size);
	cudaMalloc((void **) &d_b, size);
	cudaMalloc((void **) &d_c, size);

	HANDLE_ERROR(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice));

	elementwiseAnd<<<n_col, n_row>>>(d_a, d_b, d_c);

	cudaMemcpy(res, d_c, size, cudaMemcpyDeviceToHost);

	int loss = 0;

    for(size_t i = 0; i < n_row; i++){
		for(size_t j = 0; j < n_col; j++){
			size_t index = j + i * n_col;
			loss += abs(c[index] - res[index]);
		}
    }
	cout << loss << endl;
    
    free(a);
    free(b);
    free(c);
    free(res);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

	return 0;
}
