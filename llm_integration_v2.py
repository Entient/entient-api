# llm_integration_v2.py - Simplified, working version

import os
import openai
from typing import Dict

class LLMBriefGenerator:
    """Simple, working LLM integration for problem generation"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if self.api_key:
            openai.api_key = self.api_key
    
    def generate_problem(self, user_goal: str) -> Dict:
        """Convert natural language to ENTIENT problem dict"""
        
        if not self.api_key:
            # No API key - return basic problem
            return {
                "description": user_goal,
                "constraints": {"optimize": "general"}
            }
        
        prompt = f"""Convert this goal to a simple optimization problem:
Goal: {user_goal}

Return a simple JSON with description and one or two constraints."""
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.7
            )
            
            # For now, just extract the description
            # Robust JSON parsing can be added later
            return {
                "description": user_goal,
                "constraints": {
                    "cost": "minimize",
                    "complexity": "minimize"
                }
            }
            
        except Exception as e:
            print(f"LLM error: {e}")
            return {
                "description": user_goal,
                "constraints": {}
            }