__global__ void sssp_gather(
                       int* dist, 
                       int* newDist,                 
                       const int* column_offsets,
                       const int* row_indices,
                       int* active_nodes,
                       int num_nodes,
                       const int src,
                       int* ifFI, // Error injection flag
                       int iter // Iteration number
                       ) {
  
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < num_nodes && active_nodes[tid] == 1 && tid != src) {
    int newValue = INT_MAX;  // Initialize Phi value as infinity
    
    // Iterate over the in-edge neighbors of the current node
    for (int j = column_offsets[tid]; j < column_offsets[tid + 1]; j++) {
      int neighbor = row_indices[j];  // Get the neighbor's index
      if (dist[neighbor] == INT_MAX)
        continue;  // Skip if the neighbor's distance is still infinity
      newValue = min(newValue, dist[neighbor] + 1);  // Calculate the new value (minimum distance)
    }
    newDist[tid] = newValue;  // Store the computed new value
  }
}

__global__ void sssp_apply(
                      int* dist, 
                      int* newDist,  
                      int* changed,
                      int* active_nodes, 
                      const int num_nodes,
                      const int src,
                      int* ifFI, // Error injection flag
                      int iter // Iteration number
                      ) {
                   
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < num_nodes && active_nodes[tid] == 1 && tid != src) {
      int old_value = dist[tid];
      int new_value = min(newDist[tid], old_value);  // Apply the new distance if it's smaller than the current one

      dist[tid] = new_value;  // Update the distance to the new value
   
      // If the distance has changed, mark it as changed
      if (old_value != new_value) {
        changed[tid] = 1;
      }
  }
}

__global__ void sssp_scatter( 
                        const int* row_offsets,
                        const int* column_indices,
                        int* changed, 
                        int* active_nodes,
                        const int num_nodes,
                        int* ifFI, // Error injection flag
                        int iter // Iteration number
                        ) {
 
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  
  if (tid < num_nodes) {
    if (changed[tid] == 1) {  // If the node's distance was updated
      // Add the out-going neighbors to the list of active nodes for the next iteration
      for (int j = row_offsets[tid]; j < row_offsets[tid + 1]; j++) {
        int neighbor = column_indices[j];
        active_nodes[neighbor] = 1;  // Mark the neighbor as active
      }
    }
  }
}

void GPUSSSP(int* dist,
           const int* row_offsets, // Describes the starting position of out-edge neighbors in column_indices
           const int* column_indices, 
           const int* column_offsets, // Describes the starting position of in-edge neighbors in row_indices
           const int* row_indices,
           const int num_nodes, 
           const int num_edges,
           const int src) {

  // Allocate and initialize data
  int* active_nodes = (int*) malloc(num_nodes * sizeof(int));
  int num_active_nodes = num_nodes;
  
  const int iter_num = 10000;  // Iteration threshold
  
  int* ZeroArray = (int*) malloc(num_nodes * sizeof(int));  // Used for resetting to zero

  // Initialize active nodes and ZeroArray to 1 and 0 respectively
  for (int i = 0; i < num_nodes; i++) {
      active_nodes[i] = 1;
      ZeroArray[i] = 0;
  }

  // Error injection flag (set to INT_MAX initially)
  int ifFI = INT_MAX;

  // Allocate device memory for various arrays
  int* d_dist;
  int* d_newDist;
  int* d_row_offsets;
  int* d_column_indices;
  int* d_column_offsets;
  int* d_row_indices;
  int* d_active_nodes;
  int* d_changed;
  int* d_ifFI;
 
  // Allocate memory on the device (GPU)
  cudaMalloc((void**)&d_dist, num_nodes * sizeof(int));
  cudaMalloc((void**)&d_newDist, num_nodes * sizeof(int));
  cudaMalloc((void**)&d_row_offsets, (num_nodes + 1) * sizeof(int));
  cudaMalloc((void**)&d_column_indices, num_edges * sizeof(int));
  cudaMalloc((void**)&d_column_offsets, (num_nodes + 1) * sizeof(int));
  cudaMalloc((void**)&d_row_indices, num_edges * sizeof(int));
  cudaMalloc((void**)&d_active_nodes, num_nodes * sizeof(int));
  cudaMalloc((void**)&d_changed, num_nodes * sizeof(int));
  cudaMalloc((void**)&d_ifFI, sizeof(int));

  // Copy data from host memory to device memory
  cudaMemcpy(d_dist, dist, num_nodes * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_row_offsets, row_offsets, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_column_indices, column_indices, num_edges * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_column_offsets, column_offsets, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_row_indices, row_indices, num_edges * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_active_nodes, active_nodes, num_nodes * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_changed, ZeroArray, num_nodes * sizeof(int), cudaMemcpyHostToDevice);  // Set changed array to 0
  cudaMemcpy(d_ifFI, 0, sizeof(int), cudaMemcpyHostToDevice);  // Set ifFI to 0 (no error injection initially)
 
  int* tmp = (int*) malloc(num_nodes * sizeof(int));
  int iter;
  
  // Output files for iteration information and error injection result
  char iter_information[] = "iter_info.txt";
  FILE* f = fopen(iter_information, "w");

  char iter_information1[] = "outcome.txt";
  FILE* f1 = fopen(iter_information1, "a+");

  int block_size = 256;  // Number of threads per block
  int num_blocks = (num_nodes + block_size - 1) / block_size;  // Number of blocks required

  // Main iteration loop
  for (iter = 1; iter < iter_num && num_active_nodes > 0; iter++) {
    int flag = 0;
    fprintf(f, "%d:%d\n", iter, num_active_nodes);  // Output iteration information
    for (int i = 0; i < num_nodes; i++) {
        if (active_nodes[i] == 1) {
          if (flag == 1)
            fprintf(f, ",");
          fprintf(f, "%d", i + 1);  // Output active node index (1-based)
          flag = 1;
        }
    }
    fprintf(f, "\n");

    // Perform gather step
    sssp_gather<<<num_blocks, block_size>>>(d_dist, d_newDist, d_column_offsets, d_row_indices, d_active_nodes, num_nodes, src, d_ifFI, iter);
    cudaDeviceSynchronize();

    // Perform apply step
    sssp_apply<<<num_blocks, block_size>>>(d_dist, d_newDist, d_changed, d_active_nodes, num_nodes, src, d_ifFI, iter);
    cudaDeviceSynchronize();

    // Reset active nodes and changed flags
    cudaMemcpy(d_active_nodes, ZeroArray, num_nodes * sizeof(int), cudaMemcpyHostToDevice);

    // Perform scatter step
    sssp_scatter<<<num_blocks, block_size>>>(d_row_offsets, d_column_indices, d_changed, d_active_nodes, num_nodes, d_ifFI, iter);
    cudaDeviceSynchronize();

    // Update active nodes for the next iteration
    cudaMemcpy(active_nodes, d_active_nodes, num_nodes * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(d_changed, ZeroArray, num_nodes * sizeof(int), cudaMemcpyHostToDevice);  // Reset changed array
    
    num_active_nodes = 0;
    for (int i = 0; i < num_nodes; i++) {
      if (active_nodes[i] == 1) {
        num_active_nodes++;
      }
    }
  }

  // Copy the final result from device memory to host memory
  cudaMemcpy(dist, d_dist, num_nodes * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&ifFI, d_ifFI, sizeof(int), cudaMemcpyDeviceToHost);
  fprintf(f1, ",%d,%d", iter - 1, ifFI);  // Output the iteration count and error injection result

  fclose(f);
  fclose(f1);
  
  // Free device memory
  cudaFree(d_dist);
  cudaFree(d_newDist);
  cudaFree(d_row_offsets);
  cudaFree(d_column_indices);
  cudaFree(d_column_offsets);
  cudaFree(d_row_indices);
  cudaFree(d_active_nodes);
  cudaFree(d_changed);
  cudaFree(d_ifFI);
}
