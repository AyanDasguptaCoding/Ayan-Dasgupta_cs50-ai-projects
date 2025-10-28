import random

class Sentence:
    def __init__(self, cells, count):
        self.cells, self.count = set(cells), count

    def update(self, cell, is_mine):
        if cell in self.cells:
            self.cells.remove(cell)
            if is_mine: self.count -= 1

class MinesweeperAI:
    def __init__(self, h=8, w=8):
        self.h, self.w = h, w
        self.moves = set()
        self.mines = set()
        self.safe = set()
        self.knowledge = []

    def add_knowledge(self, cell, count):
        self.moves.add(cell)
        self.safe.add(cell)
        
        # Get unknown neighbors
        i, j = cell
        neighbors = {(x,y) for x in (i-1,i,i+1) for y in (j-1,j,j+1) 
                    if (x,y) != cell and 0<=x<self.h and 0<=y<self.w 
                    and (x,y) not in self.moves and (x,y) not in self.mines}
        
        # Add new knowledge and update
        self.knowledge.append(Sentence(neighbors, count))
        changed = True
        while changed:
            changed = False
            # Check for new safes/mines
            for s in self.knowledge:
                for cell in s.cells.copy():
                    if s.count == 0:
                        self.safe.add(cell)
                        changed = True
                    if len(s.cells) == s.count:
                        self.mines.add(cell)
                        changed = True
            # Update all sentences
            for s in self.knowledge[:]:
                for cell in self.safe: s.update(cell, False)
                for cell in self.mines: s.update(cell, True)
                if not s.cells: self.knowledge.remove(s)
            # Check subsets
            new_knowledge = []
            for s1 in self.knowledge:
                for s2 in self.knowledge:
                    if s1.cells < s2.cells:
                        new = Sentence(s2.cells-s1.cells, s2.count-s1.count)
                        if new not in self.knowledge+new_knowledge:
                            new_knowledge.append(new)
            if new_knowledge:
                self.knowledge += new_knowledge
                changed = True

    def make_safe_move(self):
        return (self.safe - self.moves).pop() if self.safe - self.moves else None

    def make_random_move(self):
        options = [(i,j) for i in range(self.h) for j in range(self.w)
                  if (i,j) not in self.moves and (i,j) not in self.mines]
        return random.choice(options) if options else None