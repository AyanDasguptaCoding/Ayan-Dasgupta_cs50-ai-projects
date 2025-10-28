import random

def transition_model(corpus, current_page, d):
    """Calculate probabilities for next page visit"""
    N = len(corpus)
    prob = {page: (1-d)/N for page in corpus}  # Base probability for all pages
    
    # Distribute link probabilities
    links = corpus[current_page]
    num_links = len(links) if links else N  # Handle pages with no links
    
    for page in links if links else corpus:  # If no links, treat as linking to all
        prob[page] += d/num_links
        
    return prob

def sample_pagerank(corpus, d, samples):
    """Estimate PageRank by simulating random surfer"""
    counts = {page: 0 for page in corpus}
    current = random.choice(list(corpus))
    
    for _ in range(samples):
        counts[current] += 1
        current = random.choices(
            *zip(*transition_model(corpus, current, d).items())
        )[0]
    
    return {page: count/samples for page, count in counts.items()}

def iterate_pagerank(corpus, d, threshold=0.001):
    """Calculate PageRank through iterative updates"""
    N = len(corpus)
    ranks = {page: 1/N for page in corpus}
    
    while True:
        new_ranks = {}
        max_diff = 0
        
        for page in corpus:
            # Sum PR from pages linking here
            incoming = sum(
                ranks[src]/len(links) if links else ranks[src]/N
                for src, links in corpus.items() 
                if not links or page in links
            )
            new_rank = (1-d)/N + d * incoming
            new_ranks[page] = new_rank
            max_diff = max(max_diff, abs(new_rank - ranks[page]))
        
        ranks = new_ranks
        if max_diff < threshold:
            break
            
    return ranks