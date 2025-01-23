__global__ void hits_gather(
                      const float* authority, 
                      const float* hub,
                      float* authority_value, 
                      float* hub_value, 
                      const int* row_offsets, // Outbound neighbors (for each node)
                      const int* column_indices, 
                      const int* column_offsets, // Inbound neighbors
                      const int* row_indices,
                      const int* active_nodes,
                      int num_nodes,
                      int* ifFI, // If an error occurred
                      int iter // Iteration number
                       ) 
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < num_nodes && active_nodes[tid] == 1) {
     
        // Compute the authority value of a page as the sum of hub values of all pages pointing to it
        int row_start = column_offsets[tid];  
        int row_end = column_offsets[tid + 1];  
        float value = 0.0f; 

        for (int j = row_start; j < row_end; j++) {
            int neighbor = row_indices[j]; 
            value += hub[neighbor]; 
        }
        authority_value[tid] = value; 

        // Compute the hub value of a page as the sum of authority values of all pages it points to
        row_start = row_offsets[tid];  
        row_end = row_offsets[tid + 1];  
        value = 0.0f; 
        
        for (int j = row_start; j < row_end; j++) {
            int neighbor = column_indices[j]; 
            value += authority[neighbor]; 
        }
        hub_value[tid] = value; 
    }
}

__global__ void hits_apply(
                      float* authority, 
                      float* hub, 
                      float* authority_value, 
                      float* hub_value,            
                      const int* active_nodes, 
                      int num_nodes, 
                      int* authority_changed,
                      int* hub_changed,
                       int* ifFI, // If an error occurred
                       int iter // Iteration number
                      ) {              
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
  
    if (tid < num_nodes && active_nodes[tid] == 1) {
        // Update authority value and check for changes
        float new_authority_value = authority_value[tid];  
        float old_authority_value = authority[tid]; 
        
        authority[tid] = new_authority_value;  
        
        float diff = fabs(new_authority_value - old_authority_value);  // Compute the difference between the new and old authority values
        if (diff >= 0.01f) {
            authority_changed[tid] = 1; 
        } 
        
        // Update hub value and check for changes
        float new_hub_value = hub_value[tid];  
        float old_hub_value = hub[tid]; 
        
        hub[tid] = new_hub_value;  
        
        diff = fabs(new_hub_value - old_hub_value);  // Compute the difference between the new and old hub values
        if (diff >= 0.01f) {
            hub_changed[tid] = 1; 
        } 
    }
}

__global__ void hits_scatter(
                        int* active_nodes, 
                        const int num_nodes,
                        const int* row_offsets, // Outbound neighbors
                        const int* column_indices, 
                        const int* column_offsets, // Inbound neighbors
                        const int* row_indices,
                        int* authority_changed,
                        int* hub_changed,
                         int* ifFI, // If an error occurred
                       int iter // Iteration number
                        ) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < num_nodes) {
        if (authority_changed[tid] == 1) {  // If authority value changed, make inbound neighbors active
            int row_start = column_offsets[tid];  
            int row_end = column_offsets[tid + 1]; 

            for (int j = row_start; j < row_end; j++) {
                int neighbor = row_indices[j];
                active_nodes[neighbor] = 1;  
            }
        }
        if (hub_changed[tid] == 1) {  // If hub value changed, make outbound neighbors active
            int row_start = row_offsets[tid];  
            int row_end = row_offsets[tid + 1]; 

            for (int j = row_start; j < row_end; j++) {
                int neighbor = column_indices[j];
                active_nodes[neighbor] = 1;  
            }
        }
    }
}

void GPUHITS(float* authority,
           float* hub,
           const int* row_offsets, // Describes the start positions of outbound neighbors in column_indices
           const int* column_indices, 
           const int* column_offsets, // Describes the start positions of inbound neighbors in row_indices
           const int* row_indices,
           const int num_nodes, 
           const int num_edges) {

    // Allocate and initialize data
    int* active_nodes =  (int*) malloc(num_nodes * sizeof(int));  // Active nodes
    int num_active_nodes = num_nodes;
  
    const int iter_num = 10000; // Maximum number of iterations

    int* ZeroArray = (int*) malloc(num_nodes * sizeof(int));  // Array used for resetting values

    for (int i = 0; i < num_nodes; i++) {
        active_nodes[i] = 1;
        ZeroArray[i] = 0;
        authority[i] = 1.0;
        hub[i] = 1.0;
    }

    // For error checking
    int ifFI = INT_MAX;
  
    // Allocate device memory for arrays
    float* d_authority;
    float* d_hub;
    float* d_authority_value;
    float* d_hub_value;

    int* d_row_offsets;
    int* d_column_indices;
    int* d_column_offsets;
    int* d_row_indices;

    int* d_active_nodes;
    int* d_authority_changed;
    int* d_hub_changed;

    int* d_ifFI;

    // Allocate space in device memory
    cudaMalloc((void**)&d_authority, num_nodes * sizeof(float));
    cudaMalloc((void**)&d_hub, num_nodes * sizeof(float));
    cudaMalloc((void**)&d_authority_value, num_nodes * sizeof(float));
    cudaMalloc((void**)&d_hub_value, num_nodes * sizeof(float));

    cudaMalloc((void**)&d_row_offsets, (num_nodes + 1) * sizeof(int));
    cudaMalloc((void**)&d_column_indices, num_edges * sizeof(int));
    cudaMalloc((void**)&d_column_offsets, (num_nodes + 1) * sizeof(int));
    cudaMalloc((void**)&d_row_indices, num_edges * sizeof(int));

    cudaMalloc((void**)&d_active_nodes, num_nodes * sizeof(int));  
    cudaMalloc((void**)&d_authority_changed, num_nodes * sizeof(int));
    cudaMalloc((void**)&d_hub_changed, num_nodes * sizeof(int));

    cudaMalloc((void**)&d_ifFI, sizeof(int));

    // Copy data from host memory to device memory
    cudaMemcpy(d_authority, authority, num_nodes * sizeof(float), cudaMemcpyHostToDevice);  
    cudaMemcpy(d_hub, hub, num_nodes * sizeof(float), cudaMemcpyHostToDevice); 

    cudaMemcpy(d_authority_value, authority, num_nodes * sizeof(int), cudaMemcpyHostToDevice); // For normalization
    cudaMemcpy(d_hub_value, hub, num_nodes * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_row_offsets, row_offsets, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_column_indices, column_indices, num_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_column_offsets, column_offsets, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_indices, row_indices, num_edges * sizeof(int), cudaMemcpyHostToDevice);
  
    cudaMemcpy(d_active_nodes, active_nodes, num_nodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hub_changed, ZeroArray, num_nodes * sizeof(int), cudaMemcpyHostToDevice); // Reset to 0
    cudaMemcpy(d_authority_changed, ZeroArray, num_nodes * sizeof(int), cudaMemcpyHostToDevice); // Reset to 0

    cudaMemcpy(d_ifFI, 0, sizeof(int), cudaMemcpyHostToDevice); // Reset to 0

    // Temporary arrays for storing values
    int* tmp = (int*) malloc(num_nodes * sizeof(int));  
    float* tmp_float = (float*) malloc(num_nodes * sizeof(float));
    int iter;

    char iter_information[] = "iter_info.txt"; // File to store iteration information
    FILE* f = fopen(iter_information, "w");

    char iter_information1[] = "outcome.txt"; // File to store results after error detection
    FILE* f1 = fopen(iter_information1, "a+");

    int block_size = 256; 
    int num_blocks = (num_nodes + block_size - 1) / block_size;

    // Iteration loop
    for (iter = 1; iter < iter_num && num_active_nodes > 0; iter++) {
        // Output iteration information
        int flag = 0;   
        fprintf(f, "%d:%d\n", iter, num_active_nodes);  
        for (int i = 0; i < num_nodes; i++) {
            if (active_nodes[i] == 1) {
                fprintf(f, "%d", i + 1);
                flag = 1;
            }
            if (i != num_nodes - 1 && flag == 1)
                fprintf(f, ",");
            flag = 0;
        }
        fprintf(f, "\n");

        // Gather phase
        hits_gather<<<num_blocks, block_size>>>(d_authority, d_hub, d_authority_value, d_hub_value, d_row_offsets, d_column_indices, d_column_offsets, d_row_indices, d_active_nodes, num_nodes, d_ifFI, iter);
        cudaDeviceSynchronize();

        // Normalize authority values
        cudaMemcpy(tmp_float, d_authority_value, num_nodes * sizeof(float), cudaMemcpyDeviceToHost);
        float authority_norm = 0;
        for (int i = 0; i < num_nodes; i++) {
            authority_norm += pow(tmp_float[i], 2);
        }
        authority_norm = sqrt(authority_norm);
        for (int i = 0; i < num_nodes; i++) {
            tmp_float[i] /= authority_norm;
        }
        cudaMemcpy(d_authority_value, tmp_float, num_nodes * sizeof(int), cudaMemcpyHostToDevice);

        // Normalize hub values
        cudaMemcpy(tmp_float, d_hub_value, num_nodes * sizeof(float), cudaMemcpyDeviceToHost);
        float hub_norm = 0;
        for (int i = 0; i < num_nodes; i++) {
            hub_norm += pow(tmp_float[i], 2);
        }
        hub_norm = sqrt(hub_norm);
        for (int i = 0; i < num_nodes; i++) {
            tmp_float[i] /= hub_norm;
        }
        cudaMemcpy(d_hub_value, tmp_float, num_nodes * sizeof(int), cudaMemcpyHostToDevice);

        // Apply phase
        hits_apply<<<num_blocks, block_size>>>(d_authority, d_hub, d_authority_value, d_hub_value, d_active_nodes, num_nodes, d_authority_changed, d_hub_changed, d_ifFI, iter);
        cudaDeviceSynchronize();
    
        // Reset all nodes to inactive before updating
        cudaMemcpy(d_active_nodes, ZeroArray, num_nodes * sizeof(int), cudaMemcpyHostToDevice);

        // Scatter phase
        hits_scatter<<<num_blocks, block_size>>>(d_active_nodes, num_nodes, d_row_offsets, d_column_indices, d_column_offsets, d_row_indices, d_authority_changed, d_hub_changed, d_ifFI, iter);
        cudaDeviceSynchronize();
    
        // Update the active nodes for the next round
        cudaMemcpy(active_nodes, d_active_nodes, num_nodes * sizeof(int), cudaMemcpyDeviceToHost);

        // Reset change flags
        cudaMemcpy(d_hub_changed, ZeroArray, num_nodes * sizeof(int), cudaMemcpyHostToDevice); // Reset to 0
        cudaMemcpy(d_authority_changed, ZeroArray, num_nodes * sizeof(int), cudaMemcpyHostToDevice); // Reset to 0
    
        num_active_nodes = 0;
        for (int i = 0; i < num_nodes; i++) {
            if (active_nodes[i] == 1) {
                num_active_nodes++;
            }
        }
    }

    // Copy results from device memory to host memory
    cudaMemcpy(authority, d_authority, num_nodes * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hub, d_hub, num_nodes * sizeof(float), cudaMemcpyDeviceToHost);
   
    cudaMemcpy(&ifFI, d_ifFI, sizeof(int), cudaMemcpyDeviceToHost);
    fprintf(f1, ",%d,%d", iter - 1, ifFI); // Output iteration number and FI status

    // Reset the device
    cudaDeviceReset(); 

    fclose(f);
    fclose(f1);

    // Free device memory
    cudaFree(d_authority);
    cudaFree(d_hub);
    cudaFree(d_authority_value);
    cudaFree(d_hub_value);

    cudaFree(d_row_offsets);
    cudaFree(d_column_indices);
    cudaFree(d_column_offsets);
    cudaFree(d_row_indices);

    cudaFree(d_active_nodes);
    cudaFree(d_authority_changed);
    cudaFree(d_hub_changed);

    cudaFree(d_ifFI);
}
