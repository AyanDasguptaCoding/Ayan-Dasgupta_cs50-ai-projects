import nltk
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP | S Conj S | S Conj VP

NP -> N | Det N | Det AdjP N | NP PP
VP -> V | V NP | V PP | Adv VP | VP Adv
PP -> P NP
AdjP -> Adj | Adj AdjP
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words. Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic character.
    """
    words = nltk.word_tokenize(sentence)
    processed = []
    for word in words:
        lower_word = word.lower()
        if any(c.isalpha() for c in lower_word):
            processed.append(lower_word)
    return processed


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    chunks = []
    for subtree in tree.subtrees():
        if subtree.label() == 'NP':
            # Check if this NP contains any other NPs
            has_nested_np = False
            for child in subtree.subtrees():
                if child != subtree and child.label() == 'NP':
                    has_nested_np = True
                    break
            if not has_nested_np:
                chunks.append(subtree)
    return chunks


def main():
    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Preprocess the sentence and parse it
    try:
        words = preprocess(s)
    except:
        print("Error: Could not preprocess input.")
        return

    try:
        trees = list(parser.parse(words))
    except ValueError as e:
        print(e)
        return
    except:
        print("Error: Could not parse input.")
        return

    if not trees:
        print("Error: Could not parse input.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


if __name__ == "__main__":
    main()