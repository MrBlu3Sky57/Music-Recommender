import numpy as np
from Graph import SongGraph
import parse
import recommend
import match


if __name__ == '__main__':
    # Load data
    vectors, identifiers, soft_data = parse.parse_file()
    normed_vectors = parse.normalize(vectors)
    ids, vector_table, name_table, soft_table = parse.build_tables(normed_vectors, identifiers, soft_data)
    id_table = {name: i for i, name in name_table.items()}
    ids = list(set(ids))

    # Query user for input
    song = ''
    while True:
        print('What is the name of a song you like?')
        name = input()
        print('What is the name of the artist(s) who made this song?')
        artist = input()

        target = (name.lower(), artist.lower())
        if target not in id_table:
            print('Sorry, we do not have an exact match for your song and artist')
            print('Did you mean one of the following?')
            candidates = match.closest(list(id_table.keys()), target)
            for i, candidate in enumerate(candidates):
                print(f"{i + 1}. {" by ".join(candidate)}")
            print("0. None of the above")
            
            while True:
                inp = input("Answer from 0 to 5: ")
                if inp in {'0', '1', '2', '3', '4', '5'}:
                    break
                else:
                    print('Error, invalid input')
            
            if int(inp) != 0:
                song = id_table[candidates[int(inp) - 1]]
                break
            else:
                print("Sorry then, let's try again")
        else:
            song = id_table[target]
            break
    
    # Generate visualizations and recommendations
    g = recommend.build_nx(ids, vector_table)
    partition, cluster_dict = recommend.partition_graph(g)
    com = partition[song]
    graph = SongGraph(np.array(list(cluster_dict[com])), vector_table)
    while True:
        print("Would you like to see how your song looks in relations to other popular songs? Yes: 1, No: any other key")
        inp = input()
        if inp == '1':
            recommend.visualize(g, vector_table, name_table, partition, song)
        break

    while True:
        print("Would you like to see how the songs in your songs community compare to it? Yes: 1, No: any other key")
        inp = input()
        if inp == '1':
            graph.visualize_heat_map(song, vector_table, soft_table, name_table)
        break

    while True:
        print("Would you like some recommendations? Yes: 1, No: any other key")
        inp = input()
        if inp == '1':
            recs = graph.recommendations(song, vector_table, soft_table, name_table)

            for rec in recs:
                print(" by ".join(rec).title())
        break
