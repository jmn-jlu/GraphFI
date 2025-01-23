#include <stdio.h> 
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>
#include"kernel.cuh"
#include"graph.cuh"
using namespace std;
#define GPU_DEVICE 0

// Structure to store value and index for sorting
struct IndexedValue {
    float value;
    int index;
};

// Comparator function to sort in descending order based on value
bool compareIndexedValue(const IndexedValue &a, const IndexedValue &b) {
    return a.value > b.value;
}

// CPU implementation of PageRank algorithm
void CPUPR(CsrGraph const &graph, float* rank)
{
  const int n = graph.nodes;
  
  // Initialize rank values to 0.15
  for (int i = 0; i < n; i++)
    rank[i] = 0.15;

  // Calculate the number of out-going edges for each node
  int* num_out_edge = (int*) malloc((n + 1) * sizeof(int));
  // Compute the difference between adjacent elements in row_offsets to get the number of out-going edges
  adjacent_difference(graph.row_offsets, graph.row_offsets + n + 1, num_out_edge);
  num_out_edge++; // Move the pointer of num_out_edge by one position

  bool changed = true;
  int iter_count = 0;

  // PageRank computation loop
  while (changed)
  {
    changed = false;
    for (int v = 0; v < n; v++)
    {
      float sumnb = 0.0;
      // Loop through the in-coming edges of node v
      for (int j = graph.column_offsets[v]; j < graph.column_offsets[v + 1]; ++j)
      {
        int nb = graph.row_indices[j]; // Neighbor of v
        sumnb += rank[nb] / (float) num_out_edge[nb];
      }
      
      // Apply the damping factor and update rank
      sumnb = 0.15 + 0.85 * sumnb;
      if (fabs(sumnb - rank[v]) >= 0.01)
      {
        changed = true; // If the rank changed, continue iterating
      }

      rank[v] = sumnb;
    }
    iter_count++;
  }
}

// L2 norm computation for two vectors
float l2norm(float* v1, float* v2, int n)
{
 float result = 0.0;
  for (unsigned int i = 0; i < n; ++i)
    result += (v1[i] - v2[i]) * (v1[i] - v2[i]);

  return sqrt(result);
}

// L2 norm computation for a single vector
float l2norm(float* v, int n)
{
  float result = 0.0;
  for (unsigned int i = 0; i < n; ++i)
    result += v[i] * v[i];

  return sqrt(result);
}

int main(int argc, char **argv)
{
 char graph_file[]= "data.mtx"; // Graph data file in Market format
 char outFileName[]="PageRank.out"; // Output file for PageRank results

    // Initialize GPU
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	cudaSetDevice( GPU_DEVICE );

    CsrGraph csr_graph; // Object representing the CSR graph
    // Load graph data from Market format file and convert to CSR (Directed graph)
    if (BuildMarketGraph(graph_file, csr_graph, false) != 0)
        return 1;

    // CPU implementation of PageRank (optional)
    int run_CPU = 0; // Control flag for CPU PageRank execution
    float* reference_ranks;
    if (run_CPU)
    {
        reference_ranks = (float*) malloc(sizeof(float) * csr_graph.nodes);
        CPUPR(csr_graph, reference_ranks);
    }
    
    // Initialize rank values for GPU PageRank
    float* rank = (float*)malloc(sizeof(float) * csr_graph.nodes);
    for(int i = 0; i < csr_graph.nodes; i++){
      rank[i] = 0.15f;
    }
    
    // Run PageRank on the GPU
    GPUPG(rank, csr_graph.row_offsets, csr_graph.column_indices, csr_graph.column_offsets, csr_graph.row_indices, csr_graph.nodes, csr_graph.edges); 

    // Compare CPU and GPU results if needed
    if(run_CPU){
        double tol = 0.1; // Tolerance for error
        printf("\nCorrectness testing ...\n"); fflush(stdout);
        const float l2error = l2norm(reference_ranks, rank, csr_graph.nodes) / l2norm(reference_ranks, csr_graph.nodes); // / sqrt((float)csr_graph.nodes);
        const bool pass = l2error < tol;
        printf("%s! l2 error = %f\n", pass ? "passed!" : "failed!", l2error);
        free(reference_ranks);
    }

    // Output the PageRank results to a file
    FILE* f = fopen(outFileName, "w");
    
    // Create a vector of indexed values (PageRank values and their respective indices)
    std::vector<IndexedValue> indexedArr;
    for (int i = 0; i < csr_graph.nodes; i++) {
        indexedArr.push_back({rank[i], i+1});
    }
    // Sort the vector in descending order based on PageRank values
    std::sort(indexedArr.begin(), indexedArr.end(), compareIndexedValue);
    
    // Write sorted results to the output file
    for (const IndexedValue &iv : indexedArr)
    {
      fprintf(f, "%d\n", iv.index);
    }
    fclose(f);
}
