import random

def transition_model(corpus, current_page, damping_factor):
    """
    Returns probability distribution of next page visit.
    With probability damping_factor, follows a random link from current page.
    With probability 1-damping_factor, jumps to a random page.
    """
    N = len(corpus)
    prob_dist = {}
    
    # Handle pages with no links (treat as linking to all pages)
    links = corpus[current_page] if corpus[current_page] else corpus.keys()
    
    # Calculate probabilities
    for page in corpus:
        prob_dist[page] = (1 - damping_factor) / N  # Random jump probability
        
        if page in links:
            prob_dist[page] += damping_factor / len(links)  # Link follow probability
            
    return prob_dist

def sample_pagerank(corpus, damping_factor, n):
    """Estimates PageRank by simulating random surfer for n samples."""
    counts = {page: 0 for page in corpus}
    current = random.choice(list(corpus))  # Start at random page
    
    for _ in range(n):
        counts[current] += 1
        model = transition_model(corpus, current, damping_factor)
        current = random.choices(list(model.keys()), weights=model.values())[0]
    
    # Normalize counts to probabilities
    return {page: count/n for page, count in counts.items()}

def iterate_pagerank(corpus, damping_factor, threshold=0.001):
    """Calculates PageRank iteratively until convergence."""
    N = len(corpus)
    ranks = {page: 1/N for page in corpus}  # Initialize equal ranks
    
    while True:
        new_ranks = {}
        max_diff = 0
        
        for page in corpus:
            # Sum PageRank from all pages that link to this page
            rank_sum = 0
            for linking_page in corpus:
                # Handle pages with no links (treat as linking to all)
                links = corpus[linking_page] if corpus[linking_page] else corpus.keys()
                if page in links:
                    rank_sum += ranks[linking_page] / len(links)
            
            # Calculate new rank
            new_rank = (1 - damping_factor)/N + damping_factor * rank_sum
            new_ranks[page] = new_rank
            max_diff = max(max_diff, abs(new_rank - ranks[page]))
        
        ranks = new_ranks
        if max_diff < threshold:  # Stop when converged
            break
            
    return ranks