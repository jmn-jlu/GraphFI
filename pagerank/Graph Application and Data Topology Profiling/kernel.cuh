__global__ void pagerank_gather(
                       float* phi, 
                       const float* rank,                    
                       const int* column_offsets, // Describes the start position of a node's incoming neighbors in the row_indices array
                       const int* row_indices,
                       const int* row_offsets,
                       const int* active_nodes,
                       int num_nodes,
                       int* ifFI, // Fault injection flag
                       int iter // Iteration number
                       ) 
{
  
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < num_nodes && active_nodes[tid] == 1) {
    int row_start = column_offsets[tid];  // Starting index of incoming neighbors
    int row_end = column_offsets[tid + 1];  // Ending index of incoming neighbors
    float phi_value = 0.0f;  // Initialize Phi value
    
    // Iterate over the current node's neighbors
    for (int j = row_start; j < row_end; j++) {
      int neighbor = row_indices[j];  // Get the index of the incoming neighbor node
      phi_value += rank[neighbor] / (float)(row_offsets[neighbor+1]-row_offsets[neighbor]);  // Compute Phi value
    }
    
    phi[tid] = phi_value;  // Store the Phi value
  }
}

__global__ void pagerank_apply(
                      float* rank, 
                      const float* phi,            
                      const int* active_nodes, 
                      int num_nodes, 
                      int* changed,
                      int* ifFI, // Fault injection flag
                      int iter // Iteration number
                      ) {              
  
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  
  if (tid < num_nodes && active_nodes[tid] == 1) {

    float new_rank_value = 0.15f + 0.85f * phi[tid];  // Compute the new rank value
    float old_rank_value = rank[tid];  // Get the old rank value
    
    rank[tid] = new_rank_value;  // Update the rank value
    
    // Compute the difference between the new and old PageRank values
    if (fabs(new_rank_value - old_rank_value) >= 0.01f) {
      changed[tid] = 1;  // Mark the node as changed
    } 
  }
}

__global__ void pagerank_scatter(
                        int* active_nodes, 
                        const int num_nodes,
                        const int* row_offsets, 
                        const int* column_indices, 
                        int* changed,
                        int* ifFI, // Fault injection flag
                        int iter // Iteration number
                        ) {
 
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  
  if (tid < num_nodes) {
    if (changed[tid] == 1) {  // If the PageRank value of the current node has changed
      int row_start = row_offsets[tid];  // Start index of outgoing neighbors
      int row_end = row_offsets[tid + 1]; // End index of outgoing neighbors

      for (int j = row_start; j < row_end; j++) {
        int neighbor = column_indices[j];
        active_nodes[neighbor] = 1;  // Mark the outgoing neighbors as active for the next iteration
        }
    }
  }
}


void GPUPG(float* rank,
           const int* row_offsets, // Describes the starting index of each node's outgoing neighbors in column_indices array
           const int* column_indices, 
           const int* column_offsets, // Describes the starting index of each node's incoming neighbors in row_indices array
           const int* row_indices,
           const int num_nodes, 
           const int num_edges) {

  // Used for fault injection
  int ifFI = INT_MAX;
  
  // Allocate and initialize data
  int* active_nodes =  (int*) malloc(num_nodes * sizeof(int));  // Active nodes
  int num_active_nodes = num_nodes;
  
  const int iter_num = 10000; // Maximum number of iterations

  int* ZeroArray = (int*) malloc(num_nodes * sizeof(int));  // Used to reset values to zero

  for (int i = 0; i < num_nodes; i++) {
      active_nodes[i] = 1;
      ZeroArray[i] = 0;
  }
 
  // Allocate device memory for arrays
  float* d_rank;
  float* d_phi;

  int* d_row_offsets;
  int* d_column_indices;
  int* d_column_offsets;
  int* d_row_indices;

  int* d_active_nodes;
  int* d_changed;

  int* d_ifFI;

  // Allocate memory on the device
  cudaMalloc((void**)&d_rank, num_nodes * sizeof(float));
  cudaMalloc((void**)&d_phi, num_nodes * sizeof(float));

  cudaMalloc((void**)&d_row_offsets, (num_nodes + 1) * sizeof(int));
  cudaMalloc((void**)&d_column_indices, num_edges * sizeof(int));
  cudaMalloc((void**)&d_column_offsets, (num_nodes + 1) * sizeof(int));
  cudaMalloc((void**)&d_row_indices, num_edges * sizeof(int));

  cudaMalloc((void**)&d_active_nodes, num_nodes * sizeof(int));  
  cudaMalloc((void**)&d_changed, num_nodes * sizeof(int));

  cudaMalloc((void**)&d_ifFI, sizeof(int));

  // Copy data from host memory to device memory
  cudaMemcpy(d_rank, rank, num_nodes * sizeof(float), cudaMemcpyHostToDevice);  // Initialize rank values to 0.15

  cudaMemcpy(d_row_offsets, row_offsets, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_column_indices, column_indices, num_edges * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_column_offsets, column_offsets, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_row_indices, row_indices, num_edges * sizeof(int), cudaMemcpyHostToDevice);
  
  cudaMemcpy(d_active_nodes, active_nodes, num_nodes * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_changed, ZeroArray, num_nodes * sizeof(int), cudaMemcpyHostToDevice); // Reset values to zero
  
  cudaMemcpy(d_ifFI, 0, sizeof(int), cudaMemcpyHostToDevice); // Reset fault injection flag
  
  int iter;

  // Output files for iteration information
  char iter_information[] = "iter_info.txt"; // Record iteration information for each run
  FILE* f = fopen(iter_information, "w");

  char iter_information1[] = "outcome.txt"; // Record iteration results after fault injection
  FILE* f1 = fopen(iter_information1, "a+");

  int block_size = 256; 
  int num_blocks = (num_nodes + block_size - 1) / block_size;

  // Iteration loop
  for (iter = 1; iter < iter_num && num_active_nodes > 0; iter++) {

    int flag = 0;   
    fprintf(f, "%d:%d\n", iter, num_active_nodes);  // Output iteration information
    for (int i = 0; i < num_nodes; i++) {
        if (active_nodes[i] == 1) {
          if (flag == 1)
          fprintf(f, ",");
          
          fprintf(f, "%d", i + 1);
          flag = 1;
        }
      }
    fprintf(f, "\n");

    // Gather phase
    pagerank_gather<<<num_blocks, block_size>>>(d_phi, d_rank, d_column_offsets, d_row_indices, d_row_offsets, d_active_nodes, num_nodes, d_ifFI, iter);
    cudaDeviceSynchronize();

    // Apply phase
    pagerank_apply<<<num_blocks, block_size>>>(d_rank, d_phi, d_active_nodes, num_nodes, d_changed, d_ifFI, iter);
    cudaDeviceSynchronize();
    
    // Scatter phase
    cudaMemcpy(d_active_nodes, ZeroArray, num_nodes * sizeof(int), cudaMemcpyHostToDevice); // Reset active nodes to inactive before updating
    
    pagerank_scatter<<<num_blocks, block_size>>>(d_active_nodes, num_nodes, d_row_offsets, d_column_indices, d_changed, d_ifFI, iter);
    cudaDeviceSynchronize();
    
    // Update active nodes for the next iteration
    cudaMemcpy(active_nodes, d_active_nodes, num_nodes * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(d_changed, ZeroArray, num_nodes * sizeof(int), cudaMemcpyHostToDevice); // Reset changed nodes
    
    num_active_nodes = 0;
    for (int i = 0; i < num_nodes; i++) {
      if (active_nodes[i] == 1) {
        num_active_nodes++;
      }
    }
  }

  // Copy the results back from device memory to host memory
  cudaMemcpy(rank, d_rank, num_nodes * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&ifFI, d_ifFI, sizeof(int), cudaMemcpyDeviceToHost);
 
  // Output the iteration count and fault injection status
  fprintf(f1, ",%d,%d", iter - 1, ifFI); // Iteration count, whether FI was successful
  
  fclose(f);
  fclose(f1);

  // Free device memory
  cudaFree(d_rank);
  cudaFree(d_phi);

  cudaFree(d_row_offsets);
  cudaFree(d_column_indices);
  cudaFree(d_column_offsets);
  cudaFree(d_row_indices);

  cudaFree(d_active_nodes);
  cudaFree(d_changed);

  cudaFree(d_ifFI);
}
