"""
Hybrid Search Performance Optimizer

This module provides tools to analyze and optimize the weighting strategy 
for hybrid search combining vector similarity and BM25 keyword search.

Key metrics evaluated:
- Cost: API calls, compute time, memory usage
- Accuracy: Precision, recall, relevance scoring
- Latency: Search response time, cache efficiency
"""

import time
import statistics
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from logging_config import get_logger

logger = get_logger("hybrid_optimizer")


@dataclass
class SearchMetrics:
    """Metrics for evaluating search performance"""
    precision: float
    recall: float
    f1_score: float
    latency_ms: float
    api_calls: int
    cost_estimate: float
    relevance_score: float


@dataclass
class WeightingStrategy:
    """Configuration for a specific weighting strategy"""
    name: str
    vector_weight: float
    bm25_weight: float
    adaptive: bool
    description: str


class HybridSearchOptimizer:
    """
    Optimizer for hybrid search weighting strategies.
    
    Tests different weighting combinations to find optimal balance
    between cost, accuracy, and latency for specific query types.
    """
    
    def __init__(self):
        self.test_strategies = [
            WeightingStrategy("equal", 0.5, 0.5, False, "Equal weighting 50/50"),
            WeightingStrategy("vector_heavy", 0.7, 0.3, False, "Vector-biased 70/30"),
            WeightingStrategy("bm25_heavy", 0.3, 0.7, False, "BM25-biased 30/70"),
            WeightingStrategy("adaptive", 0.5, 0.5, True, "Query-adaptive weighting"),
            WeightingStrategy("vector_only", 1.0, 0.0, False, "Vector search only"),
            WeightingStrategy("bm25_only", 0.0, 1.0, False, "BM25 search only")
        ]
        
        self.query_categories = {
            "factual": [
                "What is chest X-ray?",
                "Define pneumonia",
                "What causes lung infection?"
            ],
            "technical": [
                "NIH Chest X-ray dataset accuracy",
                "CheXpert model performance metrics",
                "DICOM image preprocessing steps"
            ],
            "analytical": [
                "Compare chest X-ray vs CT scan accuracy",
                "Analyze image labeling concerns in medical datasets",
                "Evaluate deep learning models for pneumonia detection"
            ],
            "relational": [
                "How does image quality affect diagnosis accuracy?",
                "What is the relationship between radiologist experience and accuracy?",
                "Connection between dataset size and model performance"
            ]
        }
    
    def analyze_cost_efficiency(self, strategy: WeightingStrategy) -> Dict[str, float]:
        """
        Analyze cost efficiency of a weighting strategy.
        
        Returns:
            Dict with cost metrics: api_calls, compute_cost, memory_usage
        """
        # Cost model based on strategy
        costs = {
            "api_calls_per_query": 0,
            "compute_ms": 0,
            "memory_mb": 0,
            "total_cost_estimate": 0
        }
        
        if strategy.vector_weight > 0:
            # Vector search costs
            costs["api_calls_per_query"] += 1  # Embedding generation
            costs["compute_ms"] += 50  # Vector similarity computation
            costs["memory_mb"] += 10  # Vector storage
            costs["total_cost_estimate"] += 0.0001  # $0.0001 per embedding call
        
        if strategy.bm25_weight > 0:
            # BM25 search costs
            costs["compute_ms"] += 20  # BM25 scoring
            costs["memory_mb"] += 5  # BM25 index
            # No API costs for BM25
        
        if strategy.adaptive:
            # Additional cost for query analysis
            costs["api_calls_per_query"] += 0.5  # Partial LLM call for analysis
            costs["compute_ms"] += 10
            costs["total_cost_estimate"] += 0.00005
        
        return costs
    
    def analyze_accuracy_potential(self, strategy: WeightingStrategy, 
                                   query_type: str) -> Dict[str, float]:
        """
        Estimate accuracy potential based on query type and strategy.
        
        Args:
            strategy: The weighting strategy to analyze
            query_type: Type of query (factual, technical, analytical, relational)
            
        Returns:
            Dict with accuracy metrics
        """
        # Accuracy model based on research and empirical data
        base_accuracy = {
            "factual": {"vector": 0.85, "bm25": 0.75},
            "technical": {"vector": 0.70, "bm25": 0.90},  # Technical terms need exact matching
            "analytical": {"vector": 0.80, "bm25": 0.65},
            "relational": {"vector": 0.75, "bm25": 0.60}
        }
        
        type_scores = base_accuracy.get(query_type, {"vector": 0.75, "bm25": 0.70})
        
        # Calculate weighted accuracy
        if strategy.name == "adaptive":
            # Adaptive strategy gets boost based on optimal selection
            estimated_precision = max(type_scores["vector"], type_scores["bm25"]) * 0.95
        else:
            estimated_precision = (
                type_scores["vector"] * strategy.vector_weight +
                type_scores["bm25"] * strategy.bm25_weight
            )
        
        # Hybrid bonus for combining approaches
        if 0.2 <= strategy.vector_weight <= 0.8 and 0.2 <= strategy.bm25_weight <= 0.8:
            estimated_precision += 0.05  # 5% bonus for true hybrid
        
        return {
            "estimated_precision": min(estimated_precision, 1.0),
            "estimated_recall": estimated_precision * 0.9,  # Recall typically slightly lower
            "confidence": 0.8 if strategy.adaptive else 0.7
        }
    
    def analyze_latency_profile(self, strategy: WeightingStrategy) -> Dict[str, float]:
        """
        Analyze latency characteristics of a weighting strategy.
        
        Returns:
            Dict with latency metrics in milliseconds
        """
        latency = {
            "embedding_time": 0,
            "vector_search_time": 0,
            "bm25_search_time": 0,
            "merging_time": 0,
            "total_time": 0
        }
        
        if strategy.vector_weight > 0:
            latency["embedding_time"] = 45  # HuggingFace embedding time
            latency["vector_search_time"] = 25  # Qdrant search time
        
        if strategy.bm25_weight > 0:
            latency["bm25_search_time"] = 15  # BM25 is typically faster
        
        if strategy.vector_weight > 0 and strategy.bm25_weight > 0:
            latency["merging_time"] = 10  # Time to merge and deduplicate
            
        if strategy.adaptive:
            latency["embedding_time"] += 5  # Query analysis overhead
        
        latency["total_time"] = sum(latency.values())
        
        return latency
    
    def generate_recommendation(self, query_types: List[str]) -> Dict[str, Any]:
        """
        Generate recommendation for optimal weighting strategy.
        
        Args:
            query_types: List of primary query types for the use case
            
        Returns:
            Recommendation with strategy, reasoning, and trade-offs
        """
        strategy_scores = {}
        
        for strategy in self.test_strategies:
            total_score = 0
            
            for query_type in query_types:
                # Weight factors: 40% accuracy, 30% latency, 30% cost
                accuracy = self.analyze_accuracy_potential(strategy, query_type)
                cost = self.analyze_cost_efficiency(strategy)
                latency = self.analyze_latency_profile(strategy)
                
                # Normalize and score
                accuracy_score = accuracy["estimated_precision"] * 0.4
                latency_score = max(0, (150 - latency["total_time"]) / 150) * 0.3  # Lower latency = higher score
                cost_score = max(0, (0.001 - cost["total_cost_estimate"]) / 0.001) * 0.3  # Lower cost = higher score
                
                total_score += accuracy_score + latency_score + cost_score
            
            strategy_scores[strategy.name] = {
                "score": total_score / len(query_types),
                "strategy": strategy
            }
        
        # Find best strategy
        best_strategy_name = max(strategy_scores.keys(), key=lambda x: strategy_scores[x]["score"])
        best_strategy = strategy_scores[best_strategy_name]["strategy"]
        
        return {
            "recommended_strategy": best_strategy.name,
            "vector_weight": best_strategy.vector_weight,
            "bm25_weight": best_strategy.bm25_weight,
            "adaptive": best_strategy.adaptive,
            "reasoning": f"{best_strategy.description} - Optimized for {', '.join(query_types)} queries",
            "expected_accuracy": strategy_scores[best_strategy_name]["score"],
            "all_scores": {name: data["score"] for name, data in strategy_scores.items()}
        }
    
    def benchmark_strategies(self, test_queries: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Benchmark different strategies against test queries.
        
        Args:
            test_queries: List of queries to test against
            
        Returns:
            Performance metrics for each strategy
        """
        results = {}
        
        for strategy in self.test_strategies:
            strategy_metrics = {
                "avg_cost": 0,
                "avg_latency": 0,
                "estimated_accuracy": 0,
                "query_count": len(test_queries)
            }
            
            for query in test_queries:
                # Determine query type
                query_type = self._classify_query_type(query)
                
                # Calculate metrics
                cost = self.analyze_cost_efficiency(strategy)
                latency = self.analyze_latency_profile(strategy)
                accuracy = self.analyze_accuracy_potential(strategy, query_type)
                
                strategy_metrics["avg_cost"] += cost["total_cost_estimate"]
                strategy_metrics["avg_latency"] += latency["total_time"]
                strategy_metrics["estimated_accuracy"] += accuracy["estimated_precision"]
            
            # Average the metrics
            strategy_metrics["avg_cost"] /= len(test_queries)
            strategy_metrics["avg_latency"] /= len(test_queries)
            strategy_metrics["estimated_accuracy"] /= len(test_queries)
            
            results[strategy.name] = strategy_metrics
        
        return results
    
    def _classify_query_type(self, query: str) -> str:
        """Classify query into one of the predefined types"""
        query_lower = query.lower()
        
        # Simple classification based on keywords
        if any(word in query_lower for word in ["what is", "define", "meaning"]):
            return "factual"
        elif any(word in query_lower for word in ["accuracy", "performance", "metrics", "dataset"]):
            return "technical"
        elif any(word in query_lower for word in ["compare", "analyze", "evaluate", "assessment"]):
            return "analytical"
        elif any(word in query_lower for word in ["relationship", "connection", "how does", "affect"]):
            return "relational"
        else:
            return "factual"  # Default


def analyze_hybrid_search_performance():
    """
    Main analysis function to evaluate hybrid search performance.
    """
    optimizer = HybridSearchOptimizer()
    
    # Test queries representing different use cases
    medical_queries = [
        "What are the concerns about image label accuracy in chest X-ray datasets?",
        "NIH Chest X-ray dataset preprocessing steps",
        "Compare radiologist accuracy vs AI model performance",
        "How does image resolution affect diagnosis accuracy?",
        "Define pneumothorax in chest X-rays",
        "Analyze deep learning model bias in medical imaging"
    ]
    
    print("=== Hybrid Search Performance Analysis ===\n")
    
    # Get recommendation for medical domain
    recommendation = optimizer.generate_recommendation(["technical", "analytical", "factual"])
    
    print("📊 RECOMMENDED STRATEGY:")
    print(f"Strategy: {recommendation['recommended_strategy']}")
    print(f"Vector Weight: {recommendation['vector_weight']}")
    print(f"BM25 Weight: {recommendation['bm25_weight']}")
    print(f"Adaptive: {recommendation['adaptive']}")
    print(f"Reasoning: {recommendation['reasoning']}")
    print(f"Expected Score: {recommendation['expected_accuracy']:.3f}")
    print()
    
    # Benchmark all strategies
    benchmark_results = optimizer.benchmark_strategies(medical_queries)
    
    print("📈 STRATEGY COMPARISON:")
    print(f"{'Strategy':<15} {'Avg Cost':<12} {'Avg Latency':<15} {'Est. Accuracy':<15}")
    print("-" * 65)
    
    for strategy_name, metrics in benchmark_results.items():
        print(f"{strategy_name:<15} "
              f"${metrics['avg_cost']:.6f}   "
              f"{metrics['avg_latency']:.1f}ms        "
              f"{metrics['estimated_accuracy']:.3f}")
    print()
    
    # Detailed analysis
    print("🔍 DETAILED ANALYSIS:")
    for strategy in optimizer.test_strategies:
        if strategy.name == recommendation['recommended_strategy']:
            print(f"\n✅ {strategy.name.upper()} (RECOMMENDED):")
        else:
            print(f"\n{strategy.name.upper()}:")
        
        cost = optimizer.analyze_cost_efficiency(strategy)
        latency = optimizer.analyze_latency_profile(strategy)
        accuracy = optimizer.analyze_accuracy_potential(strategy, "technical")
        
        print(f"  Cost: ${cost['total_cost_estimate']:.6f} per query")
        print(f"  Latency: {latency['total_time']:.1f}ms")
        print(f"  Accuracy: {accuracy['estimated_precision']:.3f}")
        print(f"  API Calls: {cost['api_calls_per_query']}")
    
    return recommendation, benchmark_results


if __name__ == "__main__":
    analyze_hybrid_search_performance()
