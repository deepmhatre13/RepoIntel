from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class QueryIntent(str, Enum):
    REPOSITORY = "REPOSITORY"
    FILE = "FILE"
    SYMBOL = "SYMBOL"
    DEPENDENCY = "DEPENDENCY"
    ARCHITECTURE = "ARCHITECTURE"
    CONFIGURATION = "CONFIGURATION"
    USAGE = "USAGE"

class ParsedQuery(BaseModel):
    raw_query: str
    intents: List[QueryIntent] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    target_files: List[str] = Field(default_factory=list)
    target_symbols: List[str] = Field(default_factory=list)

def classify_query(query: str) -> ParsedQuery:
    q_lower = query.lower()
    intents = []
    keywords = [k for k in q_lower.split() if len(k) > 2]
    
    if any(term in q_lower for term in ["dependency", "package", "framework", "library", "req", "pip", "npm"]):
        intents.append(QueryIntent.DEPENDENCY)
    if any(term in q_lower for term in ["config", "environment", "env", "settings", "yaml", "json", "properties"]):
        intents.append(QueryIntent.CONFIGURATION)
    if any(term in q_lower for term in ["arch", "structure", "design", "overview", "entrypoint", "main"]):
        intents.append(QueryIntent.ARCHITECTURE)
    if any(term in q_lower for term in ["class", "def", "function", "method", "symbol"]):
        intents.append(QueryIntent.SYMBOL)
    if any(term in q_lower for term in ["how to", "use", "example", "run", "start"]):
        intents.append(QueryIntent.USAGE)

    if not intents:
        intents.append(QueryIntent.REPOSITORY)

    return ParsedQuery(
        raw_query=query,
        intents=intents,
        keywords=keywords
    )


class QueryUnderstandingEngine:
    def analyze(self, query: str) -> Dict[str, Any]:
        pq = classify_query(query)
        return {
            "query": pq.raw_query,
            "intents": [i.value for i in pq.intents],
            "keywords": pq.keywords,
            "target_files": pq.target_files,
            "target_symbols": pq.target_symbols,
        }

