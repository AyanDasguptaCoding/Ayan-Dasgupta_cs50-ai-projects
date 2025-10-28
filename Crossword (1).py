import random
from itertools import product

class CrosswordCreator:
    def __init__(self, crossword):
        self.crossword = crossword
        self.domains = {
            var: set(self.crossword.words) for var in self.crossword.variables
        }

    def enforce_node_consistency(self):
        """Remove words from domains that don't match variable lengths"""
        for var in self.domains:
            self.domains[var] = {
                word for word in self.domains[var] 
                if len(word) == var.length
            }

    def revise(self, x, y):
        """Make x arc-consistent with y by removing conflicting words"""
        revised = False
        overlap = self.crossword.overlaps[x, y]
        
        if not overlap:
            return False
            
        i, j = overlap
        to_remove = set()
        
        for x_word in self.domains[x]:
            # Check if any y_word matches the overlap
            has_match = any(
                x_word[i] == y_word[j] 
                for y_word in self.domains[y]
            )
            if not has_match:
                to_remove.add(x_word)
                
        if to_remove:
            self.domains[x] -= to_remove
            revised = True
            
        return revised

    def ac3(self, arcs=None):
        """Enforce arc consistency using AC3 algorithm"""
        queue = arcs or [
            (x, y) for x in self.domains 
            for y in self.crossword.neighbors(x)
        ]
        
        while queue:
            x, y = queue.pop(0)
            if self.revise(x, y):
                if not self.domains[x]:
                    return False
                for z in self.crossword.neighbors(x) - {y}:
                    queue.append((z, x))
        return True

    def assignment_complete(self, assignment):
        """Check if all variables are assigned"""
        return all(var in assignment for var in self.crossword.variables)

    def consistent(self, assignment):
        """Check if current assignment is consistent"""
        # Check all words are unique
        words = list(assignment.values())
        if len(words) != len(set(words)):
            return False
            
        # Check lengths match
        for var, word in assignment.items():
            if len(word) != var.length:
                return False
                
        # Check neighbor constraints
        for var1, var2 in product(assignment, repeat=2):
            if var1 == var2:
                continue
            overlap = self.crossword.overlaps[var1, var2]
            if overlap and assignment[var1][overlap[0]] != assignment[var2][overlap[1]]:
                return False
                
        return True

    def order_domain_values(self, var, assignment):
        """Order domain values by least constraining first"""
        def count_conflicts(word):
            count = 0
            for neighbor in self.crossword.neighbors(var):
                if neighbor in assignment:
                    continue
                i, j = self.crossword.overlaps[var, neighbor]
                count += sum(1 for w in self.domains[neighbor] if w[j] != word[i])
            return count
            
        return sorted(self.domains[var], key=count_conflicts)

    def select_unassigned_variable(self, assignment):
        """Select next variable using MRV and degree heuristics"""
        unassigned = [v for v in self.crossword.variables if v not in assignment]
        # Sort by MRV, then degree
        return min(
            unassigned,
            key=lambda v: (len(self.domains[v]), -len(self.crossword.neighbors(v)))
        )

    def backtrack(self, assignment):
        """Backtracking search with inference"""
        if self.assignment_complete(assignment):
            return assignment
            
        var = self.select_unassigned_variable(assignment)
        for value in self.order_domain_values(var, assignment):
            new_assignment = assignment.copy()
            new_assignment[var] = value
            
            if self.consistent(new_assignment):
                # Try forward checking
                old_domains = {v: self.domains[v].copy() for v in self.domains}
                self.domains[var] = {value}
                
                if self.ac3(arcs=[(n, var) for n in self.crossword.neighbors(var)]):
                    result = self.backtrack(new_assignment)
                    if result:
                        return result
                
                # Restore domains if failed
                self.domains = old_domains
                
        return None

    def solve(self):
        """Solve the crossword puzzle"""
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack({})