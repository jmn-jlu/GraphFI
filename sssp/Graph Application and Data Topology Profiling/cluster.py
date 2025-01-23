import networkx as nx
import pandas as pd


def local_clustering_coefficient(graph, node):
    # Calculate the local clustering coefficient for a node
    in_neighbors = set(graph.predecessors(node))  # Incoming neighbors
    out_neighbors = set(graph.successors(node))   # Outgoing neighbors

    # Calculate the number of triangles and the total possible edges between incoming neighbors
    in_triangles = sum(
        1 for u in in_neighbors
        for v in graph.successors(u) if v in in_neighbors)
    in_possible_triangles = len(in_neighbors) * (len(in_neighbors) - 1)

    # Calculate the number of triangles and the total possible edges between outgoing neighbors
    out_triangles = sum(
        1 for v in out_neighbors
        for u in graph.predecessors(v) if u in out_neighbors)
    out_possible_triangles = len(out_neighbors) * (len(out_neighbors) - 1)

    # Calculate the local clustering coefficient for incoming and outgoing neighbors
    in_coefficient = 0 if in_possible_triangles == 0 else in_triangles / in_possible_triangles
    out_coefficient = 0 if out_possible_triangles == 0 else out_triangles / out_possible_triangles

    # Return the average of incoming and outgoing clustering coefficients
    return (in_coefficient + out_coefficient) / 2
    # Alternatively, only return the outgoing coefficient by using: return out_coefficient


def find_high_clustering_subsets(graph, threshold=0.5):
    # Find all subsets of nodes with a clustering coefficient above the given threshold
    high_clustering_subsets = []

    # Set of processed nodes to avoid reprocessing
    processed_nodes = set()

    for node in graph.nodes():
        # Skip the node if it's already processed
        if node in processed_nodes:
            continue

        # If the node's clustering coefficient is greater than or equal to the threshold, start a new subset
        if local_clustering_coefficient(graph, node) >= threshold:
            current_subset = {node}
            stack = [node]

            # Explore neighbors and add them to the current subset if they meet the condition
            while stack:
                current_node = stack.pop()
                neighbors = set(graph.successors(current_node)) | set(graph.predecessors(current_node))
                for neighbor in neighbors:
                    # If the neighbor's clustering coefficient is above the threshold and it hasn't been processed, add it
                    if local_clustering_coefficient(graph, neighbor) >= threshold and neighbor not in processed_nodes:
                        current_subset.add(neighbor)
                        stack.append(neighbor)
                        processed_nodes.add(neighbor)

            # If the subset contains more than one node, add it to the result list
            if len(current_subset) > 1:
                high_clustering_subsets.append(current_subset)
    
    return high_clustering_subsets


def save_to_excel(data, file_path):
    # Save the data (nodes and clustering coefficients) to an Excel file
    df = pd.DataFrame(data, columns=["Node", "Clustering Coefficient"])
    df.to_excel(file_path, index=False)


def read_mtx_file(file_path):
    # Read graph data from an mtx file
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Extract edges from the mtx file, ensuring that self-loops are excluded
    edges = [tuple(map(int, line.split()[:2])) for line in lines[3:] if line.split()[0] != line.split()[1]]
    graph = nx.DiGraph(edges)  # Create a directed graph
    return graph


def main():
    mtx_file_path = "GP.mtx"  # Path to the mtx file
    graph = read_mtx_file(mtx_file_path)  # Read the graph from the mtx file

    # Save nodes and their corresponding local clustering coefficients to an Excel file
    data = []

    for node in graph.nodes():
        coefficient = local_clustering_coefficient(graph, node)
        data.append({"Node": node, "Clustering Coefficient": coefficient})

    save_to_excel(data, "output.xlsx")  # Save the data to an Excel file

    # Find high clustering coefficient subsets in the graph
    high_clustering_subsets = find_high_clustering_subsets(graph)

    print("\nHigh Clustering Subsets:")
    i = 1
    for subset in high_clustering_subsets:
        print(str(i) + " " + str(subset))
        i += 1


if __name__ == "__main__":
    main()
