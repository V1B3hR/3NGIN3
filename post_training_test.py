#!/usr/bin/env python3
"""
Post-Training Performance Validation for 3NGIN3

This module runs a focused performance test after training to validate improvements.

Usage:
    python post_training_test.py
"""

import logging
import time
import json
import statistics
from performance_test import Performance3NGIN3Tester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_post_training_validation():
    """Run post-training performance validation."""
    logger.info("🔬 POST-TRAINING PERFORMANCE VALIDATION")
    logger.info("="*60)
    
    tester = Performance3NGIN3Tester()
    
    # Run focused performance tests
    tester.initialize_engine()
    
    # Quick performance check with fewer iterations
    x_results = tester.benchmark_x_axis_performance(iterations=30)
    z_results = tester.benchmark_z_axis_optimization(iterations=20)
    duet_results = tester.benchmark_duetmind_collaboration(iterations=10)
    learning_eval = tester.evaluate_learning_capabilities()
    
    # Generate comparison report
    post_training_report = {
        'x_axis_performance': x_results['summary'],
        'z_axis_performance': z_results['summary'],
        'duetmind_performance': duet_results['summary'],
        'learning_evaluation': learning_eval['overall_assessment'],
        'timestamp': time.time()
    }
    
    # Calculate post-training scores
    overall_performance = (
        x_results['summary']['overall_performance_score'] +
        z_results['summary']['overall_optimization_score'] / 10.0 +
        duet_results['summary']['overall_collaboration_score']
    ) / 3.0
    
    learning_score = (
        learning_eval['overall_assessment']['components_showing_improvement'] / 
        max(1, learning_eval['overall_assessment']['total_components_tested']) +
        min(1.0, max(0.0, learning_eval['overall_assessment']['average_learning_trend'] + 0.5))
    ) / 2.0
    
    logger.info(f"\n📊 POST-TRAINING RESULTS:")
    logger.info(f"   Performance Score: {overall_performance:.3f}")
    logger.info(f"   Learning Score: {learning_score:.3f}")
    logger.info(f"   Components Learning: {learning_eval['overall_assessment']['components_showing_improvement']}/{learning_eval['overall_assessment']['total_components_tested']}")
    
    # Save post-training results
    with open('/tmp/3ngin3_post_training_results.json', 'w') as f:
        json.dump(post_training_report, f, indent=2, default=str)
    
    return post_training_report

if __name__ == "__main__":
    run_post_training_validation()