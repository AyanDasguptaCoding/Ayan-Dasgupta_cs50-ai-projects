def joint_probability(people, one_gene, two_genes, have_trait):
    """Calculate joint probability of genetic inheritance and traits"""
    prob = 1
    
    for person in people:
        # Determine gene count (0, 1, or 2)
        if person in two_genes:
            gene_count = 2
        elif person in one_gene:
            gene_count = 1
        else:
            gene_count = 0
        
        # Calculate gene probability
        if not people[person]['mother']:  # No parents listed
            gene_prob = PROBS["gene"][gene_count]
        else:
            mother = people[person]['mother']
            father = people[person]['father']
            
            # Get parent probabilities
            mother_prob = (
                1 - PROBS["mutation"] if mother in two_genes else
                PROBS["mutation"] if mother not in one_gene and mother not in two_genes else
                0.5
            )
            
            father_prob = (
                1 - PROBS["mutation"] if father in two_genes else
                PROBS["mutation"] if father not in one_gene and father not in two_genes else
                0.5
            )
            
            # Calculate inheritance probability
            if gene_count == 2:
                gene_prob = mother_prob * father_prob
            elif gene_count == 1:
                gene_prob = mother_prob * (1 - father_prob) + (1 - mother_prob) * father_prob
            else:
                gene_prob = (1 - mother_prob) * (1 - father_prob)
        
        # Calculate trait probability
        trait = person in have_trait
        trait_prob = PROBS["trait"][gene_count][trait]
        
        # Multiply into joint probability
        prob *= gene_prob * trait_prob
    
    return prob

def update(probabilities, one_gene, two_genes, have_trait, p):
    """Update probabilities with new joint probability"""
    for person in probabilities:
        # Update gene probabilities
        if person in two_genes:
            probabilities[person]["gene"][2] += p
        elif person in one_gene:
            probabilities[person]["gene"][1] += p
        else:
            probabilities[person]["gene"][0] += p
        
        # Update trait probabilities
        probabilities[person]["trait"][person in have_trait] += p

def normalize(probabilities):
    """Normalize probability distributions to sum to 1"""
    for person in probabilities:
        # Normalize gene probabilities
        total = sum(probabilities[person]["gene"].values())
        for gene in probabilities[person]["gene"]:
            probabilities[person]["gene"][gene] /= total
        
        # Normalize trait probabilities
        total = sum(probabilities[person]["trait"].values())
        for trait in probabilities[person]["trait"]:
            probabilities[person]["trait"][trait] /= total