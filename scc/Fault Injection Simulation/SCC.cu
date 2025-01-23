#include <stdio.h> 
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>
#include <stack>
#include <set>
#include "kernel.cuh"
#include "graph.cuh"

using namespace std;
#define GPU_DEVICE 0

// Function to perform strong connectivity components (SCC) detection
void strongConnect(int v, const CsrGraph& graph, vector<int>& index, vector<int>& lowlink, vector<bool>& onStack, stack<int>& stack, int& currentIndex, int& componentCount, int* label) {
    index[v] = currentIndex;
    lowlink[v] = currentIndex;
    currentIndex++;
    stack.push(v);
    onStack[v] = true;

    // Explore all neighbors of the current node
    for (int i = graph.row_offsets[v]; i < graph.row_offsets[v + 1]; ++i) {
        const int u = graph.column_indices[i]; // Neighbor u

        if (index[u] == -1) {
            // If the neighbor is unvisited, recurse
            strongConnect(u, graph, index, lowlink, onStack, stack, currentIndex, componentCount, label);
            lowlink[v] = min(lowlink[v], lowlink[u]);
        } else if (onStack[u]) {
            // If the neighbor is on the stack, update the lowlink
            lowlink[v] = min(lowlink[v], index[u]);
        }
    }

    // If the current node is the root of an SCC
    if (lowlink[v] == index[v]) {
        int node;
        do {
            node = stack.top();
            stack.pop();
            onStack[node] = false;
            label[node] = label[v];  // Update the label of the node to the current SCC ID
        } while (node != v);

        componentCount++;  // Increment the SCC count
    }
}

// CPU implementation of SCC detection
void CPUSCC(CsrGraph const &graph, int* label)
{
    printf("Running CPU code...\n"); fflush(stdout);

    for(int i = 0; i < graph.nodes; i++){
        label[i] = i + 1; // Initialize each node with a unique label
    }

    const int vertices = graph.nodes;
    vector<int> index(vertices, -1); // Stores search order
    vector<int> lowlink(vertices, -1); // Stores SCC assignment
    vector<bool> onStack(vertices, false); // Tracks if the node is on the stack
    stack<int> stack; // Stack for DFS
    int currentIndex = 0; // Current DFS order index
    int componentCount = 0; // Number of SCCs

    for (int v = 0; v < vertices; ++v) { // Traverse each vertex
        if (index[v] == -1) {
            strongConnect(v, graph, index, lowlink, onStack, stack, currentIndex, componentCount, label);
        }
    }

    // Compute the number of distinct connected components
    { 
        std::set<int> distinctSet;
        for (int v = 0; v < vertices; v++) {
            distinctSet.insert(label[v]);
        }

        printf("CPU result: %lld connected components.\n", (long long) distinctSet.size());

        // Update labels based on component grouping
        for (int i = 0; i < vertices; i++) {
            if (label[i] > i + 1) {
                int newValue = i + 1;
                int oldValue = label[i];
                label[i] = newValue;
                for (int j = i + 1; j < vertices; j++) {
                    if (label[j] == oldValue) {
                        label[j] = newValue;
                    }
                }
            }
        }
    }
}

// Function to validate the correctness of CPU and GPU results
bool correctTest(const int nodes, int* reference_dists, int* h_dists)
{
    bool pass = true;
    int nerr = 0;
    printf("\nCorrectness testing...\n"); fflush(stdout);
    for (int i = 0; i < nodes; i++) {
        if (reference_dists[i] != h_dists[i]) {
            if (nerr++ < 20) {  // Print the first 20 errors
                printf("Incorrect value for node %d: CPU value %d, GPU value %d\n", i + 1, reference_dists[i], h_dists[i]);
            }
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
    char graph_file[] = "GP.mtx"; // Graph data file
    char outFileName[] = "SCC.out"; // Output file

    // Initialize GPU
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
    //printf("setting device %d with name %s\n", GPU_DEVICE, deviceProp.name);
    cudaSetDevice(GPU_DEVICE);

    CsrGraph csr_graph; // Object to represent CSR graph
    // Load graph data from the MARKET format file and convert it to CSR format
    if (BuildMarketGraph(graph_file, csr_graph, false) != 0)
        return 1;

    // Print CSR graph data (optional)
    //printCSR(csr_graph);

    int run_CPU = 0; // Control whether to run the CPU-based SCC algorithm
    int* reference_labels;
    if (run_CPU)
    {
        reference_labels = (int*) malloc(sizeof(int) * csr_graph.nodes);

        CPUSCC(csr_graph, reference_labels);
        /*
        printf("CPU Result: ");
        for (int i = 0; i < csr_graph.nodes; i++) {
            printf("%d ", reference_labels[i]);
        }
        printf("\n");
        */
    }

    int* label = (int*) malloc(sizeof(int) * csr_graph.nodes);
    for (int i = 0; i < csr_graph.nodes; i++) {
        label[i] = i + 1;
    }
    GPUSCC(label, csr_graph.row_offsets, csr_graph.column_indices, csr_graph.column_offsets, csr_graph.row_indices, csr_graph.nodes, csr_graph.edges); 

    // Compare GPU and CPU execution results
    if (run_CPU) {
        const bool ok = correctTest(csr_graph.nodes, reference_labels, label);    
    }

    // Output the results to a file
    if (outFileName)
    {
        FILE* f = fopen(outFileName, "w");
        for (int i = 0; i < csr_graph.nodes; ++i)
        {
            fprintf(f, "%d\n", label[i]);
        }
        fclose(f);
    }
    free(reference_labels);
}
