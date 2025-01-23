/**
 * SCC Algorithm
 * Gather Phase:
 * * For each active vertex in the current iteration, it traverses each neighbor vertex and finds the minimum connected component identifier.
 * Apply Phase:
 * * Updates the current node's identifier by taking the smaller of the neighbor node's minimum connected component identifier and the current active node's identifier.
 * * Compares the current active vertex's old and new connected component identifiers. If they are different, changed = 1, otherwise 0.
 * Scatter Phase:
 * * When a vertex's identifier changes (changed = 1), the neighbor vertices with identifiers greater than the current node are added to the active vertices for the next iteration.
 */

__global__ void scc_gather(
                       int* label,
                       int* gather,                     
                       const int* row_offsets,
                       const int* column_indices,
                       const int* column_offsets,
                       const int* row_indices,
                       bool *FWReach, 
                       bool *BWReach, 
                       int pitch1,
                       int pitch2,
                       int* active_nodes,
                       int num_nodes,
                       int* ifFI,
                       int iter) 
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < num_nodes && active_nodes[tid] == 1) {
        bool *forward = (bool *)((char *)FWReach + tid* pitch1);
        bool *backward = (bool *)((char *)BWReach + tid* pitch2);
        int gather_value = INT_MAX; 
        // Traverse the current node's neighbors (inbound and outbound edges) and find the minimum connected component identifier.
        
        for (int j = row_offsets[tid]; j < row_offsets[tid + 1]; j++) { // outbound edges
            int neighbor = column_indices[j]; 
            if (backward[neighbor] == true){
                gather_value = min(gather_value, label[neighbor]);  
            }
        }
        for (int j = column_offsets[tid]; j < column_offsets[tid + 1]; j++) {// inbound edges
            int neighbor = row_indices[j]; 
            if (forward[neighbor] == true){
                gather_value = min(gather_value, label[neighbor]);  
            }
        }

        gather[tid] = gather_value;  // Store the minimum label value from the neighbors
    }
}


__global__ void scc_apply(
                      int* label, 
                      const int* gather,
                      int* changed,            
                      const int* active_nodes, 
                      int num_nodes,
                      int* ifFI,
                      int iter) {
                   
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < num_nodes && active_nodes[tid] == 1) {
        int old_value = label[tid]; 
        int new_value = min(old_value, gather[tid]);  
        
        label[tid] = new_value;

        if (old_value != new_value)
            changed[tid] = 1;
    }
}

__global__ void scc_scatter(
                        int* label,
                        const int* row_offsets,
                        const int* column_indices,
                        const int* column_offsets,
                        const int* row_indices,
                        bool *FWReach, 
                        bool *BWReach, 
                        int pitch1,
                        int pitch2,
                        int* changed,
                        int* active_nodes,
                        int num_nodes,
                        int* ifFI,
                        int iter) {
 
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < num_nodes) {

        // When the vertex's identifier changes, add the neighbor vertices with identifiers greater than the current node to the active nodes for the next iteration
        if (changed[tid] == 1) {  
            bool *forward = (bool *)((char *)FWReach + tid* pitch1);
            bool *backward = (bool *)((char *)BWReach + tid* pitch2);

            for (int j = row_offsets[tid]; j < row_offsets[tid + 1]; j++) { // outbound edges
                int neighbor = column_indices[j]; 
                if (backward[neighbor] == true && label[neighbor] > label[tid]){
                    active_nodes[neighbor] = 1;  
                }
            }
            for (int j = column_offsets[tid]; j < column_offsets[tid + 1]; j++) {// inbound edges
                int neighbor = row_indices[j]; 
                if (forward[neighbor] == true && label[neighbor] > label[tid]){
                    active_nodes[neighbor] = 1;   
                }
            }
        }
    }
}


// Calculate all nodes' forward (FWReach) and backward (BWReach) reachability sets
void CalculateAllFWBWSets(bool **FWReach, bool **BWReach, const int *d_vertexArray, const int *d_edgeArray, int numVertices) {
     
    for(int vertex=0;vertex<numVertices;vertex++){  
        int *vertexQueue = (int*) malloc(numVertices * sizeof(int));

        // Initialize the vertex to itself as forward and backward reachable
        FWReach[vertex][vertex] = true;
        BWReach[vertex][vertex] = true;
        
        // Use a queue to traverse the outgoing neighbors of the vertex, updating the forward reachability of the vertex and backward reachability of the neighbors.
   
        int  head = 0; // head is the queue index
        vertexQueue[0] = vertex;
        while (head != -1) { // While the queue is not empty
            int currentVertex = vertexQueue[head]; // Get the current vertex
            head--;
            for(int j = d_vertexArray[currentVertex]; j < d_vertexArray[currentVertex+1]; j++) { // Neighbor node's index range
                int i = d_edgeArray[j]; // Neighbor node i
    
                if (FWReach[vertex][i] == false) {
                    FWReach[vertex][i] = true;
                    BWReach[i][vertex] = true;

                    head++;
                    vertexQueue[head] = i;
                }
            }
        }
    }
}


void GPUSCC(int* label,
           const int* row_offsets, 
           const int* column_indices, 
           const int* column_offsets, 
           const int* row_indices,
           const int num_nodes, 
           const int num_edges) {
  
    // Allocate and initialize data
    int* active_nodes =  (int*) malloc(num_nodes * sizeof(int));  // Active vertices
    int num_active_nodes = num_nodes;

    const int iter_num = 10000; // Iteration threshold

    int* ZeroArray = (int*) malloc(num_nodes * sizeof(int));  // Used for zeroing out 

    for(int i = 0; i < num_nodes; i++){
        active_nodes[i] = 1;
        ZeroArray[i] = 0;
    }

    // For fault injection
    int ifFI = INT_MAX;

    // Forward and backward reachability 2D arrays
    bool **FWReach = (bool**)malloc(num_nodes * sizeof(bool*));
    for (int i = 0; i < num_nodes; ++i) {
        FWReach[i] = (bool *)malloc(num_nodes * sizeof(bool));
    }
    bool **BWReach = (bool**)malloc(num_nodes * sizeof(bool*));
    for (int i = 0; i < num_nodes; ++i) {
        BWReach[i] = (bool *)malloc(num_nodes * sizeof(bool));
    }  

    for(int i = 0; i < num_nodes; i++) {
        for(int j = 0; j < num_nodes; j++) {
            FWReach[i][j] = false; 
            BWReach[i][j] = false;
        }
    }

    // Update the reachability arrays
    CalculateAllFWBWSets(FWReach, BWReach, row_offsets, column_indices, num_nodes); 

    // Allocate device memory for arrays
    int* d_label;
    int* d_row_offsets;
    int* d_column_indices;
    int* d_column_offsets;
    int* d_row_indices;
    int* d_active_nodes;
    int* d_changed;
    int* d_gather;
    int* d_ifFI;
    bool *d_FWReach, *d_BWReach;
    unsigned long pitch1, pitch2;

    // Allocate 2D arrays for reachability on the device
    cudaMallocPitch((void**)&d_FWReach, &pitch1, num_nodes * sizeof(bool), num_nodes );
    cudaMallocPitch((void**)&d_BWReach, &pitch2, num_nodes * sizeof(bool), num_nodes );

    // Allocate other device memory
    cudaMalloc((void**)&d_label, num_nodes * sizeof(int));
    cudaMalloc((void**)&d_row_offsets, (num_nodes + 1) * sizeof(int));
    cudaMalloc((void**)&d_column_indices, num_edges * sizeof(int));
    cudaMalloc((void**)&d_column_offsets, (num_nodes + 1) * sizeof(int));
    cudaMalloc((void**)&d_row_indices, num_edges * sizeof(int));
    cudaMalloc((void**)&d_active_nodes, num_nodes * sizeof(int));
    cudaMalloc((void**)&d_changed, num_nodes * sizeof(int));
    cudaMalloc((void**)&d_gather, num_nodes * sizeof(int));
    cudaMalloc((void**)&d_ifFI, sizeof(int));

    cudaMemcpy(d_ifFI, 0, sizeof(int), cudaMemcpyHostToDevice); // Set to 0

    // Copy data from host memory to device memory
    cudaMemcpy(d_label, label, num_nodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_offsets, row_offsets, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_column_indices, column_indices, num_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_column_offsets, column_offsets, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_indices, row_indices, num_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_active_nodes, active_nodes, num_nodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_changed, ZeroArray, num_nodes * sizeof(int), cudaMemcpyHostToDevice); // Zero out

    // Copy the reachability data to the device
    for (int i = 0; i < num_nodes; ++i) {
        cudaMemcpy2D((void*)((char*)d_FWReach + i * pitch1), pitch1,
                     (void*)FWReach[i], num_nodes * sizeof(bool),
                     num_nodes * sizeof(bool), 1, cudaMemcpyHostToDevice);
        cudaMemcpy2D((void*)((char*)d_BWReach + i * pitch2), pitch2,
                     (void*)BWReach[i], num_nodes * sizeof(bool),
                     num_nodes * sizeof(bool), 1, cudaMemcpyHostToDevice);
    }

    int iter ;
    char iter_information[] = "iter_info.txt"; // File to log iteration info
    FILE* f = fopen(iter_information, "w");

    char iter_information1[] = "outcome.txt"; // File to log fault injection results
    FILE* f1 = fopen(iter_information1, "a+");

    int block_size = 256; 
    int num_blocks = (num_nodes + block_size - 1) / block_size;

    // Iteration loop
    for (iter = 1; iter < iter_num && num_active_nodes > 0; iter++) {
        // Log iteration info
        int flag = 0;   
        fprintf(f, "%d:%d\n", iter, num_active_nodes);  
        for(int i = 0; i < num_nodes; i++){
            if(active_nodes[i] == 1){
                if(flag == 1)
                    fprintf(f, ",");
                fprintf(f, "%d", i + 1);
                flag = 1;
            }
        }
        fprintf(f, "\n");

        // Gather phase
        scc_gather<<<num_blocks, block_size>>>(d_label, d_gather, d_row_offsets, d_column_indices, d_column_offsets, d_row_indices, d_FWReach, d_BWReach, pitch1, pitch2, d_active_nodes, num_nodes, d_ifFI, iter);

        // Apply phase
        scc_apply<<<num_blocks, block_size>>>(d_label, d_gather, d_changed, d_active_nodes, num_nodes, d_ifFI, iter);

        // Scatter phase
        scc_scatter<<<num_blocks, block_size>>>(d_label, d_row_offsets, d_column_indices, d_column_offsets, d_row_indices, d_FWReach, d_BWReach, pitch1, pitch2, d_changed, d_active_nodes, num_nodes, d_ifFI, iter);
    }

    // Clean up and free allocated memory on the device
    cudaFree(d_label);
    cudaFree(d_row_offsets);
    cudaFree(d_column_indices);
    cudaFree(d_column_offsets);
    cudaFree(d_row_indices);
    cudaFree(d_active_nodes);
    cudaFree(d_changed);
    cudaFree(d_gather);
    cudaFree(d_ifFI);
    cudaFree(d_FWReach);
    cudaFree(d_BWReach);
}
