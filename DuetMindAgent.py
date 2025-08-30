import logging
from typing import Dict, Any, List
import random

logger = logging.getLogger(__name__)

class DuetMindAgent:
    """A cognitive agent with a specific style vector and reasoning capabilities."""
    
    def __init__(self, name: str, style: Dict[str, float], engine=None):
        self.name = name
        self.style = style  # e.g., {"logic": 0.9, "creativity": 0.2, "analytical": 0.8}
        self.engine = engine
        self.knowledge_graph = {}
        self.interaction_history = []
        
        logger.info(f"DuetMind Agent '{name}' initialized with style: {style}")
    
    def generate_reasoning_tree(self, task: str) -> Dict[str, Any]:
        # Use the engine's protected `safe_think` method
        result = self.engine.safe_think(self.name, task)
        
        if "error" in result:
            # The agent's reasoning was halted by the RCD
            logger.warning(f"{self.name}'s reasoning process was halted by the RCD.")
            # Create a minimal error node in the knowledge graph
            node_key = f"{self.name}_fault_{len(self.knowledge_graph)}"
            self.knowledge_graph[node_key] = {"task": task, "result": result, "style": self.style}
            return {"root": node_key, "result": result}
        
        # Process successful result into a reasoning tree
        node_key = f"{self.name}_reasoning_{len(self.knowledge_graph)}"
        
        # Apply style vector to influence reasoning
        styled_result = self._apply_style_influence(result, task)
        
        self.knowledge_graph[node_key] = {
            "task": task,
            "result": styled_result,
            "style": self.style,
            "timestamp": len(self.knowledge_graph)
        }
        
        return {
            "root": node_key,
            "result": styled_result,
            "agent": self.name,
            "style_applied": True
        }
    
    def _apply_style_influence(self, base_result: Dict[str, Any], task: str) -> Dict[str, Any]:
        """Apply the agent's style vector to influence the reasoning output."""
        styled_result = base_result.copy()
        
        # Modify based on style dimensions
        logic_weight = self.style.get("logic", 0.5)
        creativity_weight = self.style.get("creativity", 0.5)
        analytical_weight = self.style.get("analytical", 0.5)
        
        # Adjust confidence based on analytical style
        if "confidence" in styled_result:
            styled_result["confidence"] *= (0.8 + analytical_weight * 0.4)
        
        # Add style-specific insights
        style_insights = []
        
        if logic_weight > 0.7:
            style_insights.append("Applying rigorous logical validation")
        if creativity_weight > 0.7:
            style_insights.append("Exploring creative alternative perspectives")
        if analytical_weight > 0.7:
            style_insights.append("Conducting systematic decomposition")
        
        styled_result["style_insights"] = style_insights
        styled_result["style_signature"] = self.name
        
        return styled_result
    
    def dialogue_with(self, other_agent: 'DuetMindAgent', topic: str, rounds: int = 3) -> Dict[str, Any]:
        """Engage in cognitive dialogue with another agent."""
        dialogue_history = []
        current_topic = topic
        
        logger.info(f"Starting dialogue between {self.name} and {other_agent.name} on topic: {topic}")
        
        for round_num in range(rounds):
            # This agent's turn
            my_response = self.generate_reasoning_tree(f"Round {round_num + 1}: {current_topic}")
            dialogue_history.append({
                "round": round_num + 1,
                "agent": self.name,
                "response": my_response,
                "style": self.style
            })
            
            # Other agent's turn
            other_response = other_agent.generate_reasoning_tree(f"Responding to {self.name}'s perspective on: {current_topic}")
            dialogue_history.append({
                "round": round_num + 1,
                "agent": other_agent.name,
                "response": other_response,
                "style": other_agent.style
            })
            
            # Evolve the topic based on the dialogue
            current_topic = self._evolve_topic(current_topic, my_response, other_response)
        
        # Synthesize dialogue
        synthesis = self._synthesize_dialogue(dialogue_history, topic)
        
        return {
            "original_topic": topic,
            "final_topic": current_topic,
            "dialogue_history": dialogue_history,
            "synthesis": synthesis,
            "participants": [self.name, other_agent.name]
        }
    
    def _evolve_topic(self, current_topic: str, response1: Dict, response2: Dict) -> str:
        """Evolve the dialogue topic based on agent responses."""
        # Simple topic evolution based on confidence and insights
        conf1 = response1.get("result", {}).get("confidence", 0.5)
        conf2 = response2.get("result", {}).get("confidence", 0.5)
        
        if conf1 > 0.8 and conf2 > 0.8:
            return f"Deep dive into: {current_topic}"
        elif conf1 < 0.4 or conf2 < 0.4:
            return f"Alternative approach to: {current_topic}"
        else:
            return f"Balanced exploration of: {current_topic}"
    
    def _synthesize_dialogue(self, dialogue_history: List[Dict], original_topic: str) -> Dict[str, Any]:
        """Create a synthesis of the entire dialogue."""
        agent_contributions = {}
        total_confidence = 0
        insights_count = 0
        
        for entry in dialogue_history:
            agent = entry["agent"]
            if agent not in agent_contributions:
                agent_contributions[agent] = {"rounds": 0, "insights": [], "avg_confidence": 0}
            
            agent_contributions[agent]["rounds"] += 1
            
            result = entry["response"].get("result", {})
            confidence = result.get("confidence", 0.5)
            total_confidence += confidence
            
            if "style_insights" in result:
                agent_contributions[agent]["insights"].extend(result["style_insights"])
                insights_count += len(result["style_insights"])
        
        # Calculate averages
        avg_confidence = total_confidence / len(dialogue_history) if dialogue_history else 0
        
        return {
            "original_topic": original_topic,
            "dialogue_quality": avg_confidence,
            "total_insights": insights_count,
            "agent_contributions": agent_contributions,
            "cognitive_diversity": len(set(entry["agent"] for entry in dialogue_history)),
            "synthesis_summary": f"Collaborative exploration of '{original_topic}' yielding {insights_count} insights"
        }
    
    def get_agent_state(self) -> Dict[str, Any]:
        """Get current agent state and knowledge."""
        return {
            "name": self.name,
            "style": self.style,
            "knowledge_graph_size": len(self.knowledge_graph),
            "interaction_count": len(self.interaction_history),
            "engine_connected": self.engine is not None
        }
