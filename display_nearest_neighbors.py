# Author: Karl Stratos (me@karlstratos.com)
"""
This module is used to display similar words (in consine similarity).
"""
import argparse
from numpy import array
from numpy import dot
from numpy import linalg

def read_embeddings(embedding_path):
    """Reads word embeddings from various file formats."""
    embedding = {}
    dim = 0

    with open(embedding_path, "r") as embedding_file:
        line_num = 0
        for line in embedding_file:
            tokens = line.split()
            if len(tokens) > 0:
                line_num += 1

                word = tokens[0]
                starting_index = 1

                values = []
                for i in range(starting_index, len(tokens)):
                    values.append(float(tokens[i]))

                # Ensure that the dimension matches.
                if dim:
                    assert(len(values) == dim)
                else:
                    dim = len(values)

                # Set the embedding, normalize the length for later.
                embedding[word] = array(values)
                embedding[word] /= linalg.norm(embedding[word])

    return embedding, dim

def display_nearest_neighbors(args):
    """Interactively displays similar words (in consine similarity)."""
    embedding, dim = read_embeddings(args.embedding_path)
    print("Read {0} embeddings of dimension {1}".format(len(embedding), dim))

    while True:
        try:
            word = raw_input("Type a word (or just quit the program): ")
            if not word in embedding:
                print("There is no embedding for word \"{0}\"".format(word))
            else:
                neighbors = []
                for other_word in embedding:
                    if other_word == word:
                        continue
                    cosine = dot(embedding[word], embedding[other_word])
                    neighbors.append((cosine, other_word))
                neighbors.sort(reverse=True)
                for i in range(min(args.num_neighbors, len(neighbors))):
                    cosine, buddy = neighbors[i]
                    print("\t\t{0:.4f}\t\t{1}".format(cosine, buddy))
        except (KeyboardInterrupt, EOFError):
            print()
            exit(0)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("embedding_path", type=str, help="path to word "
                           "embeddings file")
    argparser.add_argument("--num_neighbors", type=int, default=30,
                           help="number of nearest neighbors to display")
    parsed_args = argparser.parse_args()
    display_nearest_neighbors(parsed_args)
