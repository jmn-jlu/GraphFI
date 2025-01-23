#include <stdio.h>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>
#include <set>
#include "kernel.cuh"
#include "graph.cuh"
using namespace std;
#define GPU_DEVICE 0

// CPU implementation of Single-Source Shortest Path (SSSP)
void CPUSSSP(CsrGraph const &graph, int* dist, int src, const bool directed)
{
    printf("Running CPU code...\n"); fflush(stdout);
    const int n = graph.nodes;
    
    // Initialize the distance array to a large number (infinity)
    for (int i = 0; i < n; i++)
        dist[i] = INT_MAX;
    
    vector<bool> visited(n);  // Vector to track visited nodes

    dist[src] = 0;  // Set the source node distance to 0

    while (true)
    {
        int u = -1;
        int sd = INT_MAX;  // Initialize shortest distance to infinity
        
        // Find the unvisited node with the smallest distance
        for (int i = 0; i < n; i++)
        {
            if (!visited[i] && dist[i] < sd)
            {
                sd = dist[i];
                u = i;
            }
        }
        
        // If no node is found, break the loop
        if (u == -1)
        {
            break;
        }
        
        visited[u] = true;  // Mark node u as visited
        
        // Forward direction: Process all outgoing edges from node u
        for (int j = graph.row_offsets[u]; j < graph.row_offsets[u + 1]; ++j)
        {
            int v = graph.column_indices[j];  // Neighbor v
            long newLen = dist[u];  // Compute new length as a long value
            newLen += 1;  // Add weight of edge (u, v)

            // Update the distance of v if a shorter path is found
            if (newLen < dist[v])
            {
                dist[v] = newLen;
            }
        }

        // If the graph is undirected, also process incoming edges
        if (!directed)
        {
            for (int j = graph.column_offsets[u]; j < graph.column_offsets[u + 1]; ++j)
            {
                int v = graph.row_indices[j];  // Neighbor v
                long newLen = dist[u];  // Compute new length as a long value
                newLen += 1;  // Add weight of edge (u, v)

                // Update the distance of v if a shorter path is found
                if (newLen < dist[v])
                {
                    dist[v] = newLen;
                }
            }
        }
    }
}

// Function to check the correctness of the GPU output by comparing it to the CPU result
bool correctTest(const int nodes, int* reference_dists, int* h_dists)
{
    bool pass = true;
    int nerr = 0;
    printf("\nCorrectness testing ...\n"); fflush(stdout);

    // Compare the CPU and GPU results
    for (int i = 0; i < nodes; i++)
    {
        if (reference_dists[i] != h_dists[i])
        {
            if (nerr++ < 20)  // Print the first 20 errors
                printf("Incorrect value for node %d: CPU value %d, GPU value %d\n", i, reference_dists[i], h_dists[i]);
            pass = false;
        }
    }
    
    if (pass)
        printf("passed\n");
    else
        printf("failed\n");
    
    return pass;
}

int main(int argc, char **argv)
{
    char graph_file[] = "data.mtx";  // Graph data file
    char outFileName[] = "SSSP.out";  // Output file for the result

    // Initialize GPU
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
    //printf("setting device %d with name %s\n", GPU_DEVICE, deviceProp.name);
    cudaSetDevice(GPU_DEVICE);

    CsrGraph csr_graph;  // Object to represent the CSR graph
    // Load the graph data from the MARKET format file and convert to CSR format
    if (BuildMarketGraph(graph_file, csr_graph, false) != 0)
        return 1;

    // Print CSR graph data
    //printCSR(csr_graph);
    int src = 135;  // Source node
    //printf("\nStarting node: %d\n", src + 1);

    int run_CPU = 0;  // Control flag for running CPU code
    int* reference_dist;
    
    // Run CPU code if needed
    if (run_CPU)
    {
        reference_dist = (int*) malloc(sizeof(int) * csr_graph.nodes);
        CPUSSSP(csr_graph, reference_dist, src, true);
        /*
        printf("CPU Result: ");
        for(int i = 0; i < csr_graph.nodes; i++){
            printf("%d ", reference_dist[i]);
        }
        printf("\n");
        */
    }

    // Execute on the GPU

    int* dist = (int*) malloc(sizeof(int) * csr_graph.nodes);
    for (int i = 0; i < csr_graph.nodes; i++)
    {
        dist[i] = INT_MAX;
    }
    dist[src] = 0;

    // Call the GPU function to run SSSP
    GPUSSSP(dist, csr_graph.row_offsets, csr_graph.column_indices, csr_graph.column_offsets, csr_graph.row_indices, csr_graph.nodes, csr_graph.edges, src);
    
    // Compare GPU and CPU results if needed
    if (run_CPU)
    {
        const bool ok = correctTest(csr_graph.nodes, reference_dist, dist);
        free(reference_dist);
    }

    // Write the result to the output file
    if (outFileName)
    {
        FILE* f = fopen(outFileName, "w");
        for (int i = 0; i < csr_graph.nodes; ++i)
        {
            fprintf(f, "%d\n", dist[i]);
        }
        fclose(f);
    }

    free(dist);
    return 0;
}
