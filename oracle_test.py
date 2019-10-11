from data import *
from parser import *

def is_inside(range_tuple, number):
    if number < range_tuple[1] and number > range_tuple[0]:
        return True
    if number > range_tuple[1] and number < range_tuple[0]:
        return True
    return False

def is_non_proj(graph_arcs):
    for i in range(len(graph_arcs)):
        for j in range(i, len(graph_arcs)):
            # all 4 cases
            # if second has one inside first and one outside 
            if is_inside(graph_arcs[i], graph_arcs[j][0]) and not is_inside(graph_arcs[i], graph_arcs[j][1]):
                return True
            elif is_inside(graph_arcs[i], graph_arcs[j][1]) and not is_inside(graph_arcs[i], graph_arcs[j][0]):
                return True
            elif is_inside(graph_arcs[j], graph_arcs[i][0]) and not is_inside(graph_arcs[j], graph_arcs[i][1]):
                return True
            elif is_inside(graph_arcs[j], graph_arcs[i][1]) and not is_inside(graph_arcs[j], graph_arcs[i][0]):
                return True
    return False

if __name__ == "__main__":
    word_embedding_path='word2vec.pkl.gz'
    if word_embedding_path.endswith('.gz'):
        with gz_open(word_embedding_path, 'rb') as file_obj:
            word_list, word_embeddings = load(file_obj)
    else:
        with open(word_embedding_path, 'rb') as file_obj:
            word_list, word_embeddings = load(file_obj)
    tag_set, deprel_set = set(), set()
    graphs = mystery.parsed_sents('train.conll')
    for graph in graphs:
        for node in graph.nodes.values():
            if node['address']:  # not root
                tag_set.add(node['ctag'])
                deprel_set.add(node['rel'])
    tag_list = sorted(tag_set)
    deprel_list = sorted(deprel_set)
    transducer = Transducer(word_list, tag_list, deprel_list)
    nonproj = 0
    total = 0
    for graph in graphs:
        # print('New Graph')
        # Exhaustive tests
        # Initialize a partial parse from the graph
        pp = PartialParse(get_sentence_from_graph(graph))
        total = total + 1
        was_nonproj = False
        # If we're not done yet
        while not pp.complete:
            # keep asking the oracle until we're
            try:
                transition_id, deprel = pp.get_oracle(graph)
                # print('Oracle says: ')
                # print(transition_id)
                # print('Sentence: ')
                # print(pp.sentence)
                # print('Stack before: ')
                # print(pp.stack)
                pp.parse_step(transition_id, deprel)
                # print('Stack after: ')
                # print(pp.stack)
            except (ValueError, IndexError):
                # no parses. If PartialParse is working, this occurs
                # when the graph is non-projective. Skip the instance
                # print(graph)
                # print('Bad Oracle! Non Proj\n')
                # print('\n\n')
                # print('Expected:\n')
                # print(list(transducer.graph2arc(graph, False)))
                # print('\nGot:\n')
                # print(pp.arcs)
                nonproj = nonproj + 1
                was_nonproj = True
                graph_arcs = list(transducer.graph2arc(graph, False))
                if not is_non_proj(graph_arcs):
                    print(graph)
                    print('Bad Non Proj!\n')
                    print('\n\n')
                    print('Expected:\n')
                    print(list(transducer.graph2arc(graph, False)))
                    print('\nGot:\n')
                    print(pp.arcs)
                    exit()
                break
        if was_nonproj:
            continue
        # Now that it's complete, let's try comparing results.
        print(transducer.deprel2id)
        pp_arcs = [(arc[0], arc[1], transducer.deprel2id[arc[2]]) for arc in pp.arcs]
        pp_arcs.sort()
        graph_arcs = list(transducer.graph2arc(graph, True))
        graph_arcs.sort()
        if pp_arcs != graph_arcs:
            print(graph)
            print('Bad Oracle!\n')
            print('\n\n')
            print('Expected:\n')
            print(list(transducer.graph2arc(graph, True)))
            print('\nGot:\n')
            print(pp_arcs)
            exit()
        print(nonproj, total, nonproj*100/total)
