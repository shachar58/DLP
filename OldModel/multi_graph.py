import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import patches
import os
import pickle


def create_multigraph(file):
    """
    this function create a multigraph out of the file given

    :param dataframe file: dataframe of the log file from the storage API calls
    :return: Multigraph g: multigraph of the log
    """

    # Preprocess data to create edge list with attributes
    aggregated_edges = {}
    for index, row in file.iterrows():
        role = row['RoleName']
        target = row['TargetName']
        operation = row['OperationUsed']
        if row['TargetType'] == "dynamodb.amazonaws.com":
            suffix = "_db"
        elif row['TargetType'] == "s3.amazonaws.com":
            suffix = "_bucket"
        read_write = "R" if row['ReadOnly'] else "W"
        edge_key = (role, target+suffix, operation, read_write)
        if edge_key in aggregated_edges:
            aggregated_edges[edge_key] += 1
        else:
            aggregated_edges[edge_key] = 1
    # Create a multi-graph using networkx
    G = nx.MultiDiGraph()
    # Add edges to the graph
    for edge_key, count in aggregated_edges.items():
        source, target, operation, read_write = edge_key
        G.add_edge(source, target, operation=operation, read_write=read_write, count=count)
    return G

# def add_permissions_toNode(G):
    # for node in G.nodes:

def plot_multigraph(G, file_path):
    """
    draw the graph given to the function, and save it to a folder

    :param Multigraph G: multigraph to draw
    :param os_filepath file_path: the path the plot will be saved to
    """
    plt.figure(figsize=(38, 18))

    # set the positions of the fixed nodes
    # fixed_positions = {"Lambda": (0, 1), "DynamoDB": (1, -1), "S3": (-1, -1)}
    node_colors = ['green' if "_bucket" in node else ('skyblue' if "_db" in node else 'orange') for node in G.nodes]
    # pos = fixed_positions.copy()
    # pos.update(nx.spring_layout(G, pos=fixed_positions, fixed=fixed_positions.keys(), scale=2))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, alpha=1)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

    ax = plt.gca()
    ax.margins(0.1)  # Add margins to the plot

    min_width = 1
    max_width = 5

    min_count = min(d["count"] for _, _, d in G.edges(data=True))
    max_count = max(d["count"] for _, _, d in G.edges(data=True))

    for u, v, key, d in G.edges(keys=True, data=True):
        edge_label = d["operation"]
        edge_count = d["count"]
        edge_color = "red" if d["read_write"] == "W" else "blue"
        edge_width = min_width + (max_width - min_width) * (edge_count - min_count) / ((max_count - min_count) + 1)
        edge_index = list(G.edges(keys=True)).index((u, v, key))

        arrow = patches.FancyArrowPatch(pos[u], pos[v],
                                        connectionstyle=f"arc3,rad={0.3 * edge_index}",
                                        arrowstyle="-", color=edge_color, linewidth=edge_width,
                                        shrinkA=5, shrinkB=5, mutation_scale=20, alpha=0.5)

        ax.add_patch(arrow)
        label_pos = arrow.get_path().vertices.mean(axis=0)
        ax.text(label_pos[0], label_pos[1], edge_label, fontsize=10, color=edge_color)

    plt.axis('off')
    # create the directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # save the figure with the key value as the filename
    plt.savefig(file_path)


if __name__ == '__main__':
    # load file into df
    file = pd.read_csv('relations_al_a.csv')
    new = file['PolicyName'] + file['Resource'] + file['Operations'] + file['Effect'] + str(file['ReadOnly'])
    new = new.unique()
    # format date into datetime
    file['SampleTime'] = pd.to_datetime(file['SampleTime'])

    # define the time window size and offset
    window_size = '10T'  # for minutes windows
    # create a dictionary of filtered dataframes
    tw_dfs = {}
    for window_range, group_df in file.groupby(
            pd.Grouper(key='SampleTime', freq=window_size)):
        tw_dfs[str(window_range)] = group_df

    # filter the dataframe based on the time windows
    tw_filter = {key: val for key, val in tw_dfs.items() if len(val) > 100}
    tw_dfs = tw_filter
    # create graphs for each time window
    graphs = {key: create_multigraph(val) for key, val in tw_dfs.items()}
    # save graph object to file
    pickle.dump(graphs, open('filename.pickle', 'wb'))

    # plot and save the graphs to a folder
    output_dir = "time_window_graphs"
    for key, val in graphs.items():
        # create filename from key, replacing special characters with '_'
        filename = key.translate(
            str.maketrans({'\\': '_', '/': '_', ':': '_', '*': '_', '?': '_', '<': '_', '>': '_', '|': '_'})) + ".png"
        file_path = os.path.join(output_dir, filename)
        plot_multigraph(val, file_path)

