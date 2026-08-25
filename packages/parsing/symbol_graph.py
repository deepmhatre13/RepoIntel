from typing import List, Dict
from packages.parsing.python_ast import CallEdge

class SymbolGraph:
    def __init__(self):
        # symbol_name -> list of CallEdge
        self.adj_out: Dict[str, List[CallEdge]] = {}
        self.adj_in: Dict[str, List[CallEdge]] = {}

    def add_edge(self, edge: CallEdge):
        src = edge.source_symbol
        tgt = edge.target_symbol
        
        if src not in self.adj_out:
            self.adj_out[src] = []
        self.adj_out[src].append(edge)

        if tgt not in self.adj_in:
            self.adj_in[tgt] = []
        self.adj_in[tgt].append(edge)

    def get_neighbors(self, symbol_or_file: str, max_neighbors: int = 10) -> List[CallEdge]:
        neighbors = []
        # Outgoing edges
        if symbol_or_file in self.adj_out:
            neighbors.extend(self.adj_out[symbol_or_file])
        # Incoming edges
        if symbol_or_file in self.adj_in:
            neighbors.extend(self.adj_in[symbol_or_file])
        
        # File-based lookup
        for edges in self.adj_out.values():
            for e in edges:
                if e.source_file == symbol_or_file or e.target_file == symbol_or_file:
                    if e not in neighbors:
                        neighbors.append(e)

        return neighbors[:max_neighbors]
