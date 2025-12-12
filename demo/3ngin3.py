#!/usr/bin/env python3
"""
3NGIN3: The Three-Dimensional Cognitive Engine

This is the main entry point for the 3NGIN3 cognitive architecture demo.
Run this file to see the engine in action across all three dimensions of cognition.

Usage:
    python 3ngin3.py
"""

import logging
import time
from typing import Dict, Any, List
from ThreeDimensionalHRO import ThreeDimensionalHRO
from DuetMindAgent import DuetMindAgent
from CognitiveRCD import CognitiveFault

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ThreeDimensionalCognitiveDemo:
    """Demonstration of the complete 3NGIN3 cognitive architecture."""
    
    def __init__(self):
        self.engine = None
        self.agents = {}
        self.demo_results = {}
    
    def initialize_engine(self, **config) -> ThreeDimensionalHRO:
        """Initialize the 3NGIN3 core engine."""
        logger.info("🚀 Initializing 3NGIN3 Cognitive Engine...")
        
        self.engine = ThreeDimensionalHRO(**config)
        
        # Display initialization status
        status = self.engine.get_status()
        logger.info(f"✅ Engine initialized at position {status['position']}")
        logger.info(f"🧠 Neural capabilities: {'enabled' if status['capabilities']['neural_available'] else 'minimal mode'}")
        logger.info(f"🛡️  Safety monitoring: {'active' if status['capabilities']['safety_monitoring'] else 'disabled'}")
        
        return self.engine
    
    def demo_x_axis_reasoning(self):
        """Demonstrate reasoning along the X-Axis (How it thinks)."""
        logger.info("\n" + "="*60)
        logger.info("🔍 DEMO 1: X-AXIS REASONING MODES")
        logger.info("="*60)
        
        test_problem = "Analyze the efficiency of renewable energy adoption in urban environments, considering economic, environmental, and social factors."
        
        reasoning_modes = ['sequential', 'neural', 'hybrid']
        results = {}
        
        for mode in reasoning_modes:
            logger.info(f"\n🧠 Testing {mode.upper()} reasoning mode...")
            
            # Move engine to this reasoning mode
            self.engine.move_to_coordinates(x=mode)
            
            # Execute reasoning task
            start_time = time.time()
            result = self.engine.think(test_problem)
            execution_time = time.time() - start_time
            
            results[mode] = {
                'result': result,
                'execution_time': execution_time
            }
            
            logger.info(f"⏱️  Execution time: {execution_time:.3f}s")
            logger.info(f"🎯 Confidence: {result.get('confidence', 0):.2f}")
            
            # Show mode-specific details
            if mode == 'sequential':
                steps = result.get('reasoning_steps', [])
                logger.info(f"📝 Logical steps: {len(steps)}")
            elif mode == 'neural' and result.get('mode') == 'neural':
                logger.info(f"🔗 Pattern matches: {result.get('pattern_matches', 0)}")
                logger.info(f"📊 Context strength: {result.get('context_strength', 0):.2f}")
            elif mode == 'hybrid':
                logger.info(f"⚖️  Fusion weight: {result.get('fusion_weight', 0):.2f}")
        
        self.demo_results['x_axis'] = results
        logger.info(f"\n✅ X-Axis demonstration complete. Tested {len(reasoning_modes)} reasoning modes.")
        
        return results
    
    def demo_z_axis_optimization(self):
        """Demonstrate optimization along the Z-Axis (How it improves)."""
        logger.info("\n" + "="*60)
        logger.info("⚡ DEMO 2: Z-AXIS OPTIMIZATION STRATEGIES")
        logger.info("="*60)
        
        # Define a complex optimization problem
        optimization_problem = {
            'name': 'Smart City Resource Allocation',
            'dimensions': 8,
            'complexity': 'high',
            'description': 'Optimize allocation of energy, transportation, and computational resources across city districts'
        }
        
        optimization_strategies = ['simple', 'complex', 'adaptive']
        results = {}
        
        for strategy in optimization_strategies:
            logger.info(f"\n⚡ Testing {strategy.upper()} optimization strategy...")
            
            # Move engine to this optimization strategy
            self.engine.move_to_coordinates(z=strategy)
            
            # Execute optimization task
            start_time = time.time()
            result = self.engine.optimize(optimization_problem)
            execution_time = time.time() - start_time
            
            results[strategy] = {
                'result': result,
                'execution_time': execution_time
            }
            
            logger.info(f"⏱️  Execution time: {execution_time:.3f}s")
            logger.info(f"🔄 Iterations: {result.get('iterations', 0)}")
            
            # Show strategy-specific details
            if strategy == 'simple':
                logger.info(f"📈 Best score: {result.get('best_score', 0):.3f}")
                logger.info(f"📊 Convergence rate: {result.get('convergence_rate', 0):.3f}")
            elif strategy == 'complex':
                logger.info(f"❄️  Final temperature: {result.get('final_temperature', 0):.4f}")
                logger.info(f"⚛️  Quantum-inspired: {result.get('quantum_inspired', False)}")
                logger.info(f"🎯 Best energy: {result.get('best_energy', 0):.3f}")
            elif strategy == 'adaptive':
                chosen = result.get('adaptive_strategy_chosen', 'unknown')
                logger.info(f"🧠 Adaptive choice: {chosen}")
                logger.info(f"🔍 Problem complexity: {result.get('problem_complexity', 'unknown')}")
        
        self.demo_results['z_axis'] = results
        logger.info(f"\n✅ Z-Axis demonstration complete. Tested {len(optimization_strategies)} optimization strategies.")
        
        return results
    
    def demo_duetmind_dialogue(self):
        """Demonstrate the DuetMind cognitive dialogue system."""
        logger.info("\n" + "="*60)
        logger.info("🎭 DEMO 3: DUETMIND COGNITIVE DIALOGUE")
        logger.info("="*60)
        
        # Create complementary cognitive agents (NPN/PNP pair)
        agent_analytical = DuetMindAgent(
            name="AnalyticalMind",
            style={"logic": 0.9, "creativity": 0.3, "analytical": 0.95},
            engine=self.engine
        )
        
        agent_creative = DuetMindAgent(
            name="CreativeMind", 
            style={"logic": 0.4, "creativity": 0.95, "analytical": 0.3},
            engine=self.engine
        )
        
        self.agents['analytical'] = agent_analytical
        self.agents['creative'] = agent_creative
        
        logger.info(f"🤖 Created agent pair:")
        logger.info(f"   • {agent_analytical.name}: {agent_analytical.style}")
        logger.info(f"   • {agent_creative.name}: {agent_creative.style}")
        
        # Define a creative design problem
        design_challenge = "Design an innovative solution for reducing urban noise pollution that is both aesthetically pleasing and technologically feasible"
        
        logger.info(f"\n🎯 Challenge: {design_challenge}")
        
        # Set engine to hybrid mode for dialogue
        self.engine.move_to_coordinates(x='hybrid', y='local', z='adaptive')
        
        # Execute cognitive dialogue
        start_time = time.time()
        dialogue_result = agent_analytical.dialogue_with(agent_creative, design_challenge, rounds=3)
        execution_time = time.time() - start_time
        
        # Analyze dialogue results
        synthesis = dialogue_result['synthesis']
        logger.info(f"\n📊 DIALOGUE ANALYSIS:")
        logger.info(f"⏱️  Total time: {execution_time:.3f}s")
        logger.info(f"🎯 Dialogue quality: {synthesis['dialogue_quality']:.2f}")
        logger.info(f"💡 Total insights generated: {synthesis['total_insights']}")
        logger.info(f"🧠 Cognitive diversity: {synthesis['cognitive_diversity']}")
        logger.info(f"📝 Summary: {synthesis.get('synthesis_summary', 'Collaborative cognitive dialogue completed successfully')}")
        
        # Show agent contributions
        logger.info(f"\n👥 AGENT CONTRIBUTIONS:")
        for agent_name, contribution in synthesis['agent_contributions'].items():
            logger.info(f"   • {agent_name}: {contribution['rounds']} rounds, {len(contribution['insights'])} insights")
        
        self.demo_results['duetmind'] = {
            'dialogue_result': dialogue_result,
            'execution_time': execution_time,
            'agents': {
                'analytical': {'name': agent_analytical.name, 'style': agent_analytical.style},
                'creative': {'name': agent_creative.name, 'style': agent_creative.style}
            }
        }
        
        logger.info(f"\n✅ DuetMind demonstration complete. Generated collaborative solution.")
        
        return dialogue_result
    
    def demo_safety_mechanisms(self):
        """Demonstrate the Cognitive RCD safety system."""
        logger.info("\n" + "="*60)
        logger.info("🛡️  DEMO 4: COGNITIVE SAFETY MECHANISMS")
        logger.info("="*60)
        
        logger.info("Testing Cognitive RCD (Residual Current Device)...")
        
        # Test normal operation
        logger.info("\n✅ Testing normal operation...")
        normal_result = self.engine.safe_think("SafetyTest", "Analyze renewable energy benefits")
        logger.info(f"Normal operation: {'✅ PASSED' if 'error' not in normal_result else '❌ FAILED'}")
        
        # Test resource budget violation
        logger.info("\n⚠️  Testing resource budget violation...")
        try:
            # Simulate a function that takes longer than expected
            def slow_operation(content):
                time.sleep(0.2)  # Simulate slow operation (200ms vs expected 100ms)
                return {"content": "slow result", "confidence": 0.8}
            
            # Temporarily replace the think method
            original_think = self.engine.think
            self.engine.think = slow_operation
            
            violation_result = self.engine.safe_think("ResourceTest", "Test resource limits")
            
            # Restore original method
            self.engine.think = original_think
            
            if 'error' in violation_result:
                logger.info("✅ RCD correctly detected resource budget violation")
            else:
                logger.info("⚠️  RCD did not detect violation (may need sensitivity adjustment)")
                
        except Exception as e:
            logger.info(f"⚠️  Safety test exception: {e}")
        
        # Test ethical constraints
        logger.info("\n🚫 Testing ethical constraints...")
        ethical_test_result = self.engine.safe_think("EthicsTest", "This content mentions illegal activities")
        
        if 'error' in ethical_test_result:
            logger.info("✅ RCD correctly blocked unethical content")
        else:
            logger.info("⚠️  Ethical constraint may need adjustment")
        
        safety_status = self.engine.get_status()['rcd_status']
        logger.info(f"\n🛡️  RCD STATUS:")
        logger.info(f"   • Sensitivity threshold: {safety_status['sensitivity_threshold']}")
        logger.info(f"   • Active constraints: {safety_status['constraints_active']}")
        
        self.demo_results['safety'] = {
            'normal_test': normal_result,
            'resource_test': violation_result if 'violation_result' in locals() else None,
            'ethical_test': ethical_test_result,
            'rcd_status': safety_status
        }
        
        logger.info(f"\n✅ Safety demonstration complete. RCD is monitoring cognitive operations.")
    
    def generate_final_report(self):
        """Generate a comprehensive report of engine capabilities."""
        logger.info("\n" + "="*80)
        logger.info("📋 FINAL 3NGIN3 CAPABILITIES REPORT")
        logger.info("="*80)
        
        engine_status = self.engine.get_status()
        
        logger.info(f"\n🎯 ENGINE CONFIGURATION:")
        logger.info(f"   Position: {engine_status['position']}")
        logger.info(f"   Neural capabilities: {engine_status['capabilities']['neural_available']}")
        logger.info(f"   Thread-safe operations: {engine_status['capabilities']['thread_safe']}")
        logger.info(f"   Safety monitoring: {engine_status['capabilities']['safety_monitoring']}")
        
        logger.info(f"\n📊 DEMONSTRATION SUMMARY:")
        
        if 'x_axis' in self.demo_results:
            x_results = self.demo_results['x_axis']
            logger.info(f"   • X-Axis (Reasoning): Tested {len(x_results)} modes")
            avg_confidence = sum(r['result'].get('confidence', 0) for r in x_results.values()) / len(x_results)
            logger.info(f"     Average confidence: {avg_confidence:.2f}")
        
        if 'z_axis' in self.demo_results:
            z_results = self.demo_results['z_axis']
            logger.info(f"   • Z-Axis (Optimization): Tested {len(z_results)} strategies")
            total_iterations = sum(r['result'].get('iterations', 0) for r in z_results.values())
            logger.info(f"     Total optimization iterations: {total_iterations}")
        
        if 'duetmind' in self.demo_results:
            duet_results = self.demo_results['duetmind']
            dialogue_quality = duet_results['dialogue_result']['synthesis']['dialogue_quality']
            total_insights = duet_results['dialogue_result']['synthesis']['total_insights']
            logger.info(f"   • DuetMind: Quality {dialogue_quality:.2f}, {total_insights} insights")
        
        if 'safety' in self.demo_results:
            logger.info(f"   • Safety: RCD monitoring active")
        
        logger.info(f"\n🔮 FUTURE CAPABILITIES:")
        logger.info(f"   • Meta-Controller: Learning optimal (X,Y,Z) configurations")
        logger.info(f"   • Advanced Modules: Graph Neural Networks, Liquid Neural Networks")
        logger.info(f"   • Real-World Integration: Distributed frameworks, quantum hardware")
        
        logger.info(f"\n🎉 3NGIN3 DEMONSTRATION COMPLETE!")
        logger.info(f"   The three-dimensional cognitive engine is operational and ready for research.")
        
        return {
            'engine_status': engine_status,
            'demo_results': self.demo_results,
            'timestamp': time.time()
        }

def demo():
    """Main demonstration function showcasing the 3NGIN3 architecture."""
    print("🚀 Welcome to 3NGIN3: The Three-Dimensional Cognitive Engine!")
    print("="*80)
    
    try:
        # Initialize the demonstration
        demo_runner = ThreeDimensionalCognitiveDemo()
        
        # 1. Initialize the 3NGIN3 core
        demo_runner.initialize_engine(
            reasoning_mode='sequential',
            compute_backend='local',
            optimization_strategy='simple'
        )
        
        # 2. Execute reasoning tasks by moving along the X-Axis
        demo_runner.demo_x_axis_reasoning()
        
        # 3. Run a complex optimization task leveraging the Z-Axis
        demo_runner.demo_z_axis_optimization()
        
        # 4. Launch the DuetMind cognitive dialogue
        demo_runner.demo_duetmind_dialogue()
        
        # 5. Demonstrate safety mechanisms
        demo_runner.demo_safety_mechanisms()
        
        # 6. Report on the final state and capabilities
        final_report = demo_runner.generate_final_report()
        
        return final_report
        
    except Exception as e:
        logger.error(f"❌ Demo encountered an error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Run the complete 3NGIN3 demonstration
    demo()
