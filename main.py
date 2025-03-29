import numpy as np
from Graph import SongGraph
import parse
import recommend


if __name__ == '__main__':
    vectors, identifiers, soft_data = parse.parse_file()
    normed_vectors = parse.normalize(vectors)
    ids, vector_table, name_table, soft_table = parse.build_tables(normed_vectors, identifiers, soft_data)
    id_table = {name: i for i, name in name_table.items()}
    ids = list(set(ids))

    g = recommend.build_nx(ids, vector_table)
    recommend.visualize(g, vector_table, name_table)
    partition, cluster_dict = recommend.partition_graph(g)
    song = id_table[('die with a smile', 'lady gaga bruno mars')]
    com = partition[song]
    graph = SongGraph(np.array(list(cluster_dict[com])), vector_table)
    graph.visualize_heat_map(song, vector_table, soft_table, name_table)
    print(graph.recommendations(song, vector_table, soft_table, name_table))
    
    # res = np.linspace(1, 2, 10)
    # thresh = np.linspace(0.5, 1.0, 10)
    # thresh_max = 0.95
    # mods = np.zeros((10))
    # g = build_nx(ids, vector_table, thresh_max)
    # for i in range(len(res)):
    #     partition, cluster_dict = partition_graph(g, 1.0)
    #     mods[i] = community.modularity(partition, g, weight='weight')
    
    # plt.plot(res, mods)
    # plt.show()