class DuetMindAgent:
    # ...
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
            
        # ... (rest of the function if successful) ...
