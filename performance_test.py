#!/usr/bin/env python3
"""
3NGIN3 Performance Testing and Learning Evaluation Framework

This module tests the performance characteristics and learning capabilities
of the 3NGIN3 cognitive engine across all three dimensions.

Usage:
    python performance_test.py
"""

import logging
import time
import statistics
import json
from typing import Dict, Any, List, Tuple
from collections import defaultdict
import random

from ThreeDimensionalHRO import ThreeDimensionalHRO
from DuetMindAgent import DuetMindAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Performance3NGIN3Tester:
    """Comprehensive performance testing and learning evaluation for 3NGIN3."""
    
    def __init__(self):
        self.engine = None
        self.test_results = {}
        self.learning_history = defaultdict(list)
        
    def initialize_engine(self, **config) -> ThreeDimensionalHRO:
        """Initialize the 3NGIN3 engine for testing."""
        logger.info("🚀 Initializing 3NGIN3 Engine for Performance Testing...")
        self.engine = ThreeDimensionalHRO(**config)
        status = self.engine.get_status()
        logger.info(f"✅ Engine initialized at {status['position']}")
        logger.info(f"🧠 Neural: {status['capabilities']['neural_available']}")
        logger.info(f"🛡️  Safety: {status['capabilities']['safety_monitoring']}")
        return self.engine
    
    def benchmark_x_axis_performance(self, iterations: int = 100) -> Dict[str, Any]:
        """Benchmark X-axis reasoning performance and learning."""
        logger.info("\n" + "="*60)
        logger.info("🔍 BENCHMARKING X-AXIS REASONING PERFORMANCE")
        logger.info("="*60)
        
        test_problems = [
            "Analyze renewable energy efficiency in urban environments considering economic factors",
            "Design algorithms for distributed computing with fault tolerance requirements",
            "Evaluate machine learning model performance across different optimization strategies",
            "Optimize resource allocation in multi-agent systems with competing objectives"
        ]
        
        reasoning_modes = ['sequential', 'neural', 'hybrid']
        results = {}
        
        for mode in reasoning_modes:
            logger.info(f"\n🧠 Testing {mode.upper()} reasoning mode...")
            self.engine.move_to_coordinates(x=mode)
            
            mode_results = {
                'execution_times': [],
                'confidence_scores': [],
                'performance_metrics': [],
                'learning_progression': []
            }
            
            # Run multiple iterations to measure performance consistency and learning
            for i in range(iterations):
                problem = random.choice(test_problems)
                
                start_time = time.perf_counter()
                result = self.engine.think(problem)
                execution_time = time.perf_counter() - start_time
                
                mode_results['execution_times'].append(execution_time)
                mode_results['confidence_scores'].append(result.get('confidence', 0.0))
                
                # Track learning progression every 10 iterations
                if i % 10 == 0:
                    avg_confidence = statistics.mean(mode_results['confidence_scores'][-10:])
                    avg_time = statistics.mean(mode_results['execution_times'][-10:])
                    mode_results['learning_progression'].append({
                        'iteration': i,
                        'avg_confidence': avg_confidence,
                        'avg_execution_time': avg_time
                    })
                    self.learning_history[f'x_axis_{mode}'].append(avg_confidence)
            
            # Calculate performance statistics
            mode_results['performance_metrics'] = {
                'mean_execution_time': statistics.mean(mode_results['execution_times']),
                'std_execution_time': statistics.stdev(mode_results['execution_times']),
                'mean_confidence': statistics.mean(mode_results['confidence_scores']),
                'std_confidence': statistics.stdev(mode_results['confidence_scores']),
                'confidence_improvement': self._calculate_learning_trend(mode_results['confidence_scores']),
                'performance_stability': 1.0 / (1.0 + statistics.stdev(mode_results['execution_times']))
            }
            
            results[mode] = mode_results
            
            logger.info(f"⏱️  Mean execution time: {mode_results['performance_metrics']['mean_execution_time']:.4f}s")
            logger.info(f"🎯 Mean confidence: {mode_results['performance_metrics']['mean_confidence']:.3f}")
            logger.info(f"📈 Learning trend: {mode_results['performance_metrics']['confidence_improvement']:.3f}")
            logger.info(f"🎖️  Performance stability: {mode_results['performance_metrics']['performance_stability']:.3f}")
        
        results['summary'] = self._summarize_x_axis_performance(results)
        self.test_results['x_axis_benchmark'] = results
        return results
    
    def benchmark_z_axis_optimization(self, iterations: int = 50) -> Dict[str, Any]:
        """Benchmark Z-axis optimization performance and adaptive learning."""
        logger.info("\n" + "="*60)
        logger.info("⚡ BENCHMARKING Z-AXIS OPTIMIZATION PERFORMANCE")
        logger.info("="*60)
        
        optimization_problems = [
            {'name': 'Resource Allocation', 'dimensions': 5, 'complexity': 'medium'},
            {'name': 'Network Routing', 'dimensions': 8, 'complexity': 'high'},
            {'name': 'Task Scheduling', 'dimensions': 3, 'complexity': 'low'},
            {'name': 'Portfolio Optimization', 'dimensions': 12, 'complexity': 'high'}
        ]
        
        optimization_strategies = ['simple', 'complex', 'adaptive']
        results = {}
        
        for strategy in optimization_strategies:
            logger.info(f"\n⚡ Testing {strategy.upper()} optimization strategy...")
            self.engine.move_to_coordinates(z=strategy)
            
            strategy_results = {
                'execution_times': [],
                'iteration_counts': [],
                'optimization_scores': [],
                'convergence_rates': [],
                'learning_progression': []
            }
            
            for i in range(iterations):
                problem = random.choice(optimization_problems)
                
                start_time = time.perf_counter()
                result = self.engine.optimize(problem)
                execution_time = time.perf_counter() - start_time
                
                strategy_results['execution_times'].append(execution_time)
                strategy_results['iteration_counts'].append(result.get('iterations', 0))
                
                # Extract optimization score based on strategy
                if strategy == 'simple':
                    score = result.get('best_score', 0.0)
                elif strategy == 'complex':
                    score = -result.get('best_energy', 0.0)  # Convert energy to positive score
                else:  # adaptive
                    score = result.get('best_score', -result.get('best_energy', 0.0))
                
                strategy_results['optimization_scores'].append(score)
                strategy_results['convergence_rates'].append(result.get('convergence_rate', 0.0))
                
                # Track learning progression
                if i % 5 == 0:
                    avg_score = statistics.mean(strategy_results['optimization_scores'][-5:])
                    avg_time = statistics.mean(strategy_results['execution_times'][-5:])
                    strategy_results['learning_progression'].append({
                        'iteration': i,
                        'avg_score': avg_score,
                        'avg_execution_time': avg_time
                    })
                    self.learning_history[f'z_axis_{strategy}'].append(avg_score)
            
            # Calculate performance metrics
            strategy_results['performance_metrics'] = {
                'mean_execution_time': statistics.mean(strategy_results['execution_times']),
                'mean_iterations': statistics.mean(strategy_results['iteration_counts']),
                'mean_optimization_score': statistics.mean(strategy_results['optimization_scores']),
                'score_improvement': self._calculate_learning_trend(strategy_results['optimization_scores']),
                'convergence_efficiency': statistics.mean(strategy_results['convergence_rates']),
                'computational_efficiency': statistics.mean(strategy_results['iteration_counts']) / statistics.mean(strategy_results['execution_times'])
            }
            
            results[strategy] = strategy_results
            
            logger.info(f"⏱️  Mean execution time: {strategy_results['performance_metrics']['mean_execution_time']:.4f}s")
            logger.info(f"🔄 Mean iterations: {strategy_results['performance_metrics']['mean_iterations']:.1f}")
            logger.info(f"🎯 Mean score: {strategy_results['performance_metrics']['mean_optimization_score']:.3f}")
            logger.info(f"📈 Score improvement: {strategy_results['performance_metrics']['score_improvement']:.3f}")
        
        results['summary'] = self._summarize_z_axis_performance(results)
        self.test_results['z_axis_benchmark'] = results
        return results
    
    def benchmark_duetmind_collaboration(self, iterations: int = 20) -> Dict[str, Any]:
        """Benchmark DuetMind collaboration performance and learning."""
        logger.info("\n" + "="*60)
        logger.info("🎭 BENCHMARKING DUETMIND COLLABORATION PERFORMANCE")
        logger.info("="*60)
        
        collaboration_tasks = [
            "Design sustainable urban transportation system",
            "Create innovative educational technology platform",
            "Develop ethical AI governance framework",
            "Plan efficient renewable energy grid",
            "Design collaborative workspace for remote teams"
        ]
        
        # Create agent pairs with different cognitive styles
        agent_pairs = [
            (DuetMindAgent("Analytical", {"logic": 0.9, "creativity": 0.3}, self.engine),
             DuetMindAgent("Creative", {"logic": 0.3, "creativity": 0.9}, self.engine)),
            (DuetMindAgent("Balanced", {"logic": 0.7, "creativity": 0.7}, self.engine),
             DuetMindAgent("Innovative", {"logic": 0.5, "creativity": 0.8}, self.engine)),
            (DuetMindAgent("Systematic", {"logic": 0.8, "creativity": 0.4}, self.engine),
             DuetMindAgent("Intuitive", {"logic": 0.4, "creativity": 0.8}, self.engine))
        ]
        
        results = {}
        
        for pair_idx, (agent1, agent2) in enumerate(agent_pairs):
            pair_name = f"pair_{pair_idx + 1}"
            logger.info(f"\n🤖 Testing agent pair {pair_idx + 1}: {agent1.name} & {agent2.name}")
            
            pair_results = {
                'execution_times': [],
                'dialogue_qualities': [],
                'insight_counts': [],
                'cognitive_diversities': [],
                'learning_progression': []
            }
            
            for i in range(iterations):
                task = random.choice(collaboration_tasks)
                
                start_time = time.perf_counter()
                dialogue_result = agent1.dialogue_with(agent2, task, rounds=3)
                execution_time = time.perf_counter() - start_time
                
                synthesis = dialogue_result['synthesis']
                
                pair_results['execution_times'].append(execution_time)
                pair_results['dialogue_qualities'].append(synthesis['dialogue_quality'])
                pair_results['insight_counts'].append(synthesis['total_insights'])
                pair_results['cognitive_diversities'].append(synthesis['cognitive_diversity'])
                
                # Track learning progression
                if i % 3 == 0:
                    avg_quality = statistics.mean(pair_results['dialogue_qualities'][-3:])
                    avg_insights = statistics.mean(pair_results['insight_counts'][-3:])
                    pair_results['learning_progression'].append({
                        'iteration': i,
                        'avg_quality': avg_quality,
                        'avg_insights': avg_insights
                    })
                    self.learning_history[f'duetmind_{pair_name}'].append(avg_quality)
            
            # Calculate performance metrics
            pair_results['performance_metrics'] = {
                'mean_execution_time': statistics.mean(pair_results['execution_times']),
                'mean_dialogue_quality': statistics.mean(pair_results['dialogue_qualities']),
                'mean_insight_count': statistics.mean(pair_results['insight_counts']),
                'quality_improvement': self._calculate_learning_trend(pair_results['dialogue_qualities']),
                'collaboration_efficiency': statistics.mean(pair_results['insight_counts']) / statistics.mean(pair_results['execution_times'])
            }
            
            results[pair_name] = pair_results
            
            logger.info(f"⏱️  Mean execution time: {pair_results['performance_metrics']['mean_execution_time']:.4f}s")
            logger.info(f"🎯 Mean dialogue quality: {pair_results['performance_metrics']['mean_dialogue_quality']:.3f}")
            logger.info(f"💡 Mean insights: {pair_results['performance_metrics']['mean_insight_count']:.1f}")
            logger.info(f"📈 Quality improvement: {pair_results['performance_metrics']['quality_improvement']:.3f}")
        
        results['summary'] = self._summarize_duetmind_performance(results)
        self.test_results['duetmind_benchmark'] = results
        return results
    
    def evaluate_learning_capabilities(self) -> Dict[str, Any]:
        """Evaluate the engine's learning capabilities across all dimensions."""
        logger.info("\n" + "="*60)
        logger.info("🧠 EVALUATING LEARNING CAPABILITIES")
        logger.info("="*60)
        
        learning_evaluation = {}
        
        # Analyze learning trends for each component
        for component, history in self.learning_history.items():
            if len(history) > 3:
                learning_trend = self._calculate_learning_trend(history)
                consistency = 1.0 / (1.0 + statistics.stdev(history))
                
                learning_evaluation[component] = {
                    'learning_trend': learning_trend,
                    'performance_consistency': consistency,
                    'improvement_rate': (history[-1] - history[0]) / len(history) if len(history) > 1 else 0.0,
                    'data_points': len(history)
                }
                
                logger.info(f"📊 {component}: trend={learning_trend:.3f}, consistency={consistency:.3f}")
        
        # Overall learning assessment
        overall_learning = {
            'components_showing_improvement': sum(1 for eval_data in learning_evaluation.values() if eval_data['learning_trend'] > 0.01),
            'total_components_tested': len(learning_evaluation),
            'average_learning_trend': statistics.mean([eval_data['learning_trend'] for eval_data in learning_evaluation.values()]),
            'average_consistency': statistics.mean([eval_data['performance_consistency'] for eval_data in learning_evaluation.values()])
        }
        
        learning_evaluation['overall_assessment'] = overall_learning
        self.test_results['learning_evaluation'] = learning_evaluation
        
        logger.info(f"\n🎯 LEARNING ASSESSMENT:")
        logger.info(f"   Components showing improvement: {overall_learning['components_showing_improvement']}/{overall_learning['total_components_tested']}")
        logger.info(f"   Average learning trend: {overall_learning['average_learning_trend']:.3f}")
        logger.info(f"   Average consistency: {overall_learning['average_consistency']:.3f}")
        
        return learning_evaluation
    
    def run_comprehensive_performance_test(self) -> Dict[str, Any]:
        """Run comprehensive performance and learning evaluation."""
        logger.info("🚀 Starting Comprehensive 3NGIN3 Performance and Learning Evaluation")
        logger.info("="*80)
        
        start_time = time.time()
        
        # Initialize engine
        self.initialize_engine()
        
        # Run all benchmarks
        self.benchmark_x_axis_performance(iterations=50)
        self.benchmark_z_axis_optimization(iterations=30)
        self.benchmark_duetmind_collaboration(iterations=15)
        
        # Evaluate learning
        self.evaluate_learning_capabilities()
        
        total_time = time.time() - start_time
        
        # Generate final report
        final_report = self._generate_final_performance_report(total_time)
        self.test_results['final_report'] = final_report
        
        logger.info("\n" + "="*80)
        logger.info("🎉 COMPREHENSIVE PERFORMANCE EVALUATION COMPLETE!")
        logger.info("="*80)
        
        return self.test_results
    
    def _calculate_learning_trend(self, values: List[float]) -> float:
        """Calculate learning trend using linear regression slope."""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x_values = list(range(n))
        
        # Simple linear regression slope calculation
        sum_x = sum(x_values)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(x_values, values))
        sum_x2 = sum(x * x for x in x_values)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        return slope
    
    def _summarize_x_axis_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize X-axis performance results."""
        best_mode = min(results.keys(), key=lambda k: results[k]['performance_metrics']['mean_execution_time'])
        most_confident = max(results.keys(), key=lambda k: results[k]['performance_metrics']['mean_confidence'])
        best_learner = max(results.keys(), key=lambda k: results[k]['performance_metrics']['confidence_improvement'])
        
        return {
            'fastest_mode': best_mode,
            'most_confident_mode': most_confident,
            'best_learning_mode': best_learner,
            'overall_performance_score': statistics.mean([
                results[mode]['performance_metrics']['performance_stability'] for mode in results.keys()
            ])
        }
    
    def _summarize_z_axis_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize Z-axis performance results."""
        best_strategy = max(results.keys(), key=lambda k: results[k]['performance_metrics']['mean_optimization_score'])
        most_efficient = max(results.keys(), key=lambda k: results[k]['performance_metrics']['computational_efficiency'])
        best_learner = max(results.keys(), key=lambda k: results[k]['performance_metrics']['score_improvement'])
        
        return {
            'best_optimization_strategy': best_strategy,
            'most_efficient_strategy': most_efficient,
            'best_learning_strategy': best_learner,
            'overall_optimization_score': statistics.mean([
                results[strategy]['performance_metrics']['mean_optimization_score'] for strategy in results.keys()
            ])
        }
    
    def _summarize_duetmind_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize DuetMind performance results."""
        best_pair = max(results.keys(), key=lambda k: results[k]['performance_metrics']['mean_dialogue_quality'])
        most_insightful = max(results.keys(), key=lambda k: results[k]['performance_metrics']['mean_insight_count'])
        best_learner = max(results.keys(), key=lambda k: results[k]['performance_metrics']['quality_improvement'])
        
        return {
            'best_collaboration_pair': best_pair,
            'most_insightful_pair': most_insightful,
            'best_learning_pair': best_learner,
            'overall_collaboration_score': statistics.mean([
                results[pair]['performance_metrics']['mean_dialogue_quality'] for pair in results.keys()
            ])
        }
    
    def _generate_final_performance_report(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive final performance report."""
        x_summary = self.test_results['x_axis_benchmark']['summary']
        z_summary = self.test_results['z_axis_benchmark']['summary']
        duet_summary = self.test_results['duetmind_benchmark']['summary']
        learning_eval = self.test_results['learning_evaluation']['overall_assessment']
        
        overall_score = (
            x_summary['overall_performance_score'] +
            z_summary['overall_optimization_score'] / 10.0 +  # Normalize to 0-1 range
            duet_summary['overall_collaboration_score']
        ) / 3.0
        
        learning_score = (
            learning_eval['components_showing_improvement'] / max(1, learning_eval['total_components_tested']) +
            min(1.0, max(0.0, learning_eval['average_learning_trend'] + 0.5))  # Normalize trend to 0-1
        ) / 2.0
        
        report = {
            'total_execution_time': total_time,
            'overall_performance_score': overall_score,
            'overall_learning_score': learning_score,
            'performance_grade': self._calculate_grade(overall_score),
            'learning_grade': self._calculate_grade(learning_score),
            'recommendations': self._generate_recommendations(overall_score, learning_score),
            'summary_metrics': {
                'x_axis_best': x_summary,
                'z_axis_best': z_summary,
                'duetmind_best': duet_summary,
                'learning_assessment': learning_eval
            }
        }
        
        logger.info(f"\n📊 FINAL PERFORMANCE REPORT:")
        logger.info(f"   Overall Performance Score: {overall_score:.3f} ({report['performance_grade']})")
        logger.info(f"   Overall Learning Score: {learning_score:.3f} ({report['learning_grade']})")
        logger.info(f"   Total Execution Time: {total_time:.2f}s")
        logger.info(f"   Is Good Learner? {'YES' if learning_score > 0.6 else 'NEEDS IMPROVEMENT'}")
        
        return report
    
    def _calculate_grade(self, score: float) -> str:
        """Convert numerical score to letter grade."""
        if score >= 0.9:
            return "A+"
        elif score >= 0.8:
            return "A"
        elif score >= 0.7:
            return "B+"
        elif score >= 0.6:
            return "B"
        elif score >= 0.5:
            return "C+"
        elif score >= 0.4:
            return "C"
        else:
            return "D"
    
    def _generate_recommendations(self, performance_score: float, learning_score: float) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        if performance_score < 0.7:
            recommendations.append("Consider optimizing reasoning algorithms for better performance")
        
        if learning_score < 0.6:
            recommendations.append("Implement adaptive learning mechanisms to improve over time")
            recommendations.append("Add memory systems to retain and apply learned patterns")
        
        if performance_score > 0.8 and learning_score > 0.7:
            recommendations.append("Excellent performance and learning! Consider expanding to more complex scenarios")
        
        if len(recommendations) == 0:
            recommendations.append("Good performance overall. Continue monitoring and fine-tuning")
        
        return recommendations

def main():
    """Main function to run performance tests."""
    tester = Performance3NGIN3Tester()
    results = tester.run_comprehensive_performance_test()
    
    # Save results to file
    with open('/tmp/3ngin3_performance_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\n💾 Results saved to /tmp/3ngin3_performance_results.json")

if __name__ == "__main__":
    main()