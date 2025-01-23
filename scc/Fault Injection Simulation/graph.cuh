#include <math.h>
#include <stdio.h>
#include <algorithm>
/**
 * Store edges represented in Coordinate (COO) format. 
 * The COO format represents each edge as a tuple (row, column, value).
 */
typedef struct CooEdgeTuple {
	int row;
	int col;
	float val;

	CooEdgeTuple(int row, int col, float val) : row(row), col(col), val(val) {}

	void Val(float &value)
	{
		val = value;
	}
}CooEdgeTuple;

/**
 * Comparator for sorting COO sparse format edges
 */
bool TupleCompare (CooEdgeTuple elem1,CooEdgeTuple elem2)
{
	if (elem1.row < elem2.row) {
		return true;
	} 
	return false;
}
bool TupleCompare2 (   CooEdgeTuple elem1,CooEdgeTuple elem2)
{
	if (elem1.col < elem2.col) {
	    return true;
	} 
	return false;
}

/**
 * Represents a graph data structure in Compressed Sparse Row (CSR) format
 */
typedef struct CsrGraph
{
    int nodes;             // Number of nodes
    int edges;             // Number of edges
    
    int *row_offsets;      // Array storing row offset values, describing the starting position of outgoing edges for each node in the column_indices array.
    int *column_indices;   // Array storing column indices, used to describe the target node for each edge.
    
    int *column_offsets;   // Array storing column offset values. For directed graphs, it describes the starting position of incoming edges for each node in the row_indices array.
    int *row_indices;      // Array storing row indices. For directed graphs, it describes the source node for each incoming edge.
    
    int *edge_values;      // Array storing the weight or attribute values of edges.


    /**
     * Constructor
     */
    CsrGraph()
    {
      nodes = 0;
      edges = 0;
      row_offsets = NULL;
      column_indices = NULL;
      edge_values = NULL;
    }

    void FromCoo(CooEdgeTuple *coo, int coo_nodes, int coo_edges, int undirected)
    {
     // printf("Converting %d vertices, %d directed edges to CSR format... ", coo_nodes, coo_edges);
      
      // Fill the CSR object
      this->nodes = coo_nodes;
      this->edges = coo_edges;

      row_offsets = (int*) malloc(sizeof(int) * (coo_nodes + 1));
      column_indices = (int*) malloc(sizeof(int) * coo_edges);
      if (!undirected) // For directed graph, store incoming edges
        {
          column_offsets = (int*) malloc(sizeof(int) * (coo_nodes + 1));
          row_indices = (int*) malloc(sizeof(int) * coo_edges);
        }

      edge_values = (int*) malloc(sizeof(int) * coo_edges) ;
    
        // Sort COO by row in ascending order
        std::stable_sort(coo, coo + coo_edges, TupleCompare);

        int prev_row = -1;
        for (int edge = 0; edge < coo_edges; edge++) // edge represents row offset
        {

          int current_row = coo[edge].row;
          // Fill row offsets up to the current row
          for (int row = prev_row + 1; row <= current_row; row++)
          {
            row_offsets[row] = edge;
          }
          prev_row = current_row;

          column_indices[edge] = coo[edge].col;
    
          edge_values[edge] = coo[edge].val;
//            coo[edge].Val(edge_values[edge]);
        }

        // Fill out any trailing nodes without edges (and the end-of-list element)
        for (int row = prev_row + 1; row <= nodes; row++)
        {
          row_offsets[row] = edges;
        }


        if (!undirected) // Directed graph
        {
          // Sort COO by column
          std::stable_sort(coo, coo + coo_edges, TupleCompare2);

          int prev_col = -1;
          for (int edge = 0; edge < edges; edge++)
          {

            int current_col = coo[edge].col;
            // Fill column offsets up to and including the current column
            for (int col = prev_col + 1; col <= current_col; col++)
            {
              column_offsets[col] = edge;
            }
            prev_col = current_col;

            row_indices[edge] = coo[edge].row;
          }
          // Fill out any trailing nodes without edges (and the end-of-list element)
          for (int col = prev_col + 1; col <= nodes; col++)
          {
            column_offsets[col] = edges;
          }
        }
      


    }
   
       
  
}CsrGraph;


/**
 * Print CSR graph data
 */
void printCSR(CsrGraph &csr_graph){
    printf("crs_graph:nodes %d, edges  %d  \n",csr_graph.nodes,csr_graph.edges);
    printf("crs_graph:row_offsets : ");
    for(int i=0;i<csr_graph.nodes+1;i++){
        printf("%d  ",csr_graph.row_offsets[i]);
    }
    printf("\ncrs_graph:column_indices : ");
    for(int i=0;i<csr_graph.edges;i++){
        printf("%d  ",csr_graph.column_indices[i]);
    }
    if(csr_graph.column_offsets != NULL){
    printf("\ncrs_graph:column_offsets : ");
    for(int i=0;i<csr_graph.nodes+1;i++){
    printf("%d  ",csr_graph.column_offsets[i]);
    }
    }
    if(csr_graph.row_indices != NULL){
    printf("\ncrs_graph:row_indices : ");
    for(int i=0;i<csr_graph.edges;i++){
    printf("%d  ",csr_graph.row_indices[i]);
    } 
    }
    printf("\n");     
    }
 


/**
 * Read graph data from a file in MARKET format and convert it into the Compressed Sparse Row (CSR) format.
 */
int BuildMarketGraph(char *graph_filename, CsrGraph &csr_graph,bool undirected)
{ 
    // Read from file
    FILE *f_in = fopen(graph_filename, "r");
    char line[1024];
    int edges_read = -1; // Number of edges read
    int nodes = 0;
    int edges = 0;
    CooEdgeTuple *coo = NULL;
    if (f_in) {
        // printf("Reading from %s:\n", graph_filename);
        // Loop to read graph file
        while(true) {
            // Read a line from file stream f_in and store it in the string line. Then it checks if the line is successfully read.
            if (fscanf(f_in, "%[^\n]\n", line) <= 0) {
               break; 
            }
            if (line[0] == '%') {
              // Comment, ignore
            }else if(edges_read == -1){   // Read the first line: dimensions (nodes and columns) and number of edges

                long long ll_nodes_x, ll_nodes_y, ll_edges; // Store the values extracted from the line
                sscanf(line, "%lld %lld %lld", &ll_nodes_x, &ll_nodes_y, &ll_edges);
                nodes = ll_nodes_x;
                edges = (undirected) ? ll_edges * 2 : ll_edges; // If the graph is undirected, add reverse edges.
                
                // Allocate memory for COO graph
                coo = (CooEdgeTuple*) malloc(sizeof(CooEdgeTuple) * edges);
                edges_read++;
            }else{  // Read edges
                long long ll_row, ll_col;
                double edge_value = 1; // Default value if edge value is not provided in the line
                int nread = sscanf(line, "%lld %lld %lf", &ll_row, &ll_col, &edge_value);
           
                coo[edges_read].row = ll_row - 1;	// zero-based array
                coo[edges_read].col = ll_col - 1;	// zero-based array
                coo[edges_read].val =(float) edge_value;

                edges_read++;

                if (undirected) {  // If the graph is undirected, add reverse edges.
                  coo[edges_read].row = ll_col - 1;	// zero-based array
                  coo[edges_read].col = ll_row - 1;	// zero-based array
                  coo[edges_read].val = (float)edge_value;
                  edges_read++;
			          }
            }
        }
      
        // Convert COO to CSR
        csr_graph.FromCoo(coo,nodes,edges,undirected);
        free(coo);
	      fflush(stdout);
        return 0;
    } else {
        perror("Unable to open file");
        return -1;
    }
}
