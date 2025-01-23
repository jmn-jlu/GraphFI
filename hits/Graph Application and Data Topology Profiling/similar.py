def parse_graph_file(file_path):
    """
    Reads graph data from a file and returns two adjacency list representations of the graph:
    one for outgoing edges and another for incoming edges.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
        num_vertices, _, num_edges = map(int, lines[0].split())
        edges = [(int(line.split()[0]), int(line.split()[1])) for line in lines[1:]]
        
    # Initialize the adjacency lists for outgoing and incoming edges
    adjacency_list_out = {i: set() for i in range(1, num_vertices + 1)}
    adjacency_list_in = {i: set() for i in range(1, num_vertices + 1)}

    # Populate the adjacency lists based on directed edges
    for v1, v2 in edges:
        adjacency_list_out[v1].add(v2)
        adjacency_list_in[v2].add(v1)

    return adjacency_list_out, adjacency_list_in

def calculate_similarity(adjacency_list_out, adjacency_list_in):
    """
    Calculates and prints the neighbor similarity for each pair of vertices,
    considering both incoming and outgoing neighbors.
    """
    num_vertices = len(adjacency_list_out)
    similarity_data = {}

    for v1 in range(1, num_vertices + 1):
        for v2 in range(v1 + 1, num_vertices + 1):
            # Combine incoming and outgoing neighbors for both vertices
            intersection_out = adjacency_list_out[v1].intersection(adjacency_list_out[v2])  # intersection ∩
            union_out = adjacency_list_out[v1].union(adjacency_list_out[v2])  # union ∪

            intersection_in = adjacency_list_in[v1].intersection(adjacency_list_in[v2])  # intersection ∩
            union_in = adjacency_list_in[v1].union(adjacency_list_in[v2])  # union ∪
            
            if union_out and union_in:  # Avoid division by zero
                similarity_out = len(intersection_out) / len(union_out)
                similarity_in = len(intersection_in) / len(union_in)
                threshold_in = 0.6
                threshold_out = 0.6
                if similarity_in >= threshold_in and similarity_out >= threshold_out:
                    similarity_data[(v1, v2)] = (round(similarity_in, 2), round(similarity_out, 2))

    print("threshold_in=" + str(threshold_in) + " and threshold_out=" + str(threshold_out))            
    return similarity_data

def construct_group(vertex, similarity_list, current_group, visited):
    """
    Recursively constructs groups based on similarity between vertices.
    """
    # Check if the vertex has similarity with other vertices in the group
    if len(current_group) != 0:
        for i in current_group:
            if vertex not in similarity_list[i]:
                return
    current_group.append(vertex)
    visited.add(vertex)
    for neighbor in similarity_list[vertex]:
        if neighbor not in visited:
            construct_group(neighbor, similarity_list, current_group, visited)

def group_vertices(similarity_data):
    """
    Groups vertices based on high similarity, ensuring that all vertices within a group 
    have high similarity with each other.
    """
    # Construct an adjacency list of the graph based on similarity data
    similarity_ori = {}
    for key, value in similarity_data.items():
        similarity_ori.setdefault(key[0], []).append(key[1])
        similarity_ori.setdefault(key[1], []).append(key[0])

    similarity_list = {k: similarity_ori[k] for k in sorted(similarity_ori)}

    visited = set()
    groups = []
    for vertex in similarity_list.keys():
        if vertex not in visited:
            for neighbor in similarity_list[vertex]:
                current_group = []
                current_group.append(vertex)
                construct_group(neighbor, similarity_list, current_group, set())
                groups.append(current_group)
    return groups

# Assuming the graph data is stored in 'graph_data.txt'
file_path = 'FPC.mtx'

# Parse the graph data from the file for a directed graph, obtaining both incoming and outgoing adjacency lists
adjacency_list_out, adjacency_list_in = parse_graph_file(file_path)

# Calculate and create a neighbor similarity list for the directed graph, considering both incoming and outgoing edges
similarity_data = calculate_similarity(adjacency_list_out, adjacency_list_in)
#print(similarity_data)

# Group vertices based on similarity
groups = group_vertices(similarity_data)
groups_tuples = [tuple(sorted(group)) for group in groups]

# Use a set to remove duplicate elements
unique_groups = set(groups_tuples)

sorted_groups = sorted(unique_groups)

# # Output the results
# for i, group in enumerate(sorted_groups):
#     print(f"Group {i + 1}: {group}")

original_list = sorted_groups
result_set = set()

# Loop until the original list is empty
num = 1

while original_list:
    # Find the longest subset containing the current number
    longest_subset = None
    for subset in original_list:
        if num in subset:
            if longest_subset is None or len(subset) > len(longest_subset):
                longest_subset = subset
    
    # If the longest subset is found
    if longest_subset:
        result_set.add(longest_subset)
        # Build the set of numbers to delete, which are all the numbers in the longest subset
        delete_numbers = set(longest_subset)
        # Remove the longest subset's numbers from the original list
        for i, sub in enumerate(original_list):
            original_list[i] = tuple(num for num in sub if num not in delete_numbers)
        original_list = [sub for sub in original_list if len(sub) > 1]

    num += 1

result_set = sorted(result_set)
for i, group in enumerate(result_set):
    print(f"Group {i + 1}: {group}")
