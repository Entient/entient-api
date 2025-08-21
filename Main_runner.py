# main_runner.py

import asyncio
import sys
import os
from typing import Dict

# Import ENTIENT components
try:
    from entient_ultimate_system import CompleteSystem, MetaAgentHarness
    from llm_integration_v2 import LLMBriefGenerator
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Ensure entient_ultimate_system.py and llm_integration_v2.py are in the same directory")
    sys.exit(1)

async def run_entient_with_prompt(user_goal: str):
    """Run ENTIENT with a natural language prompt as input"""
    
    try:
        # Step 1: Initialize ENTIENT system
        print("\n" + "="*60)
        print("Initializing ENTIENT v7.0 System")
        print("="*60)
        
        system = CompleteSystem(num_agents=6)
        harness = MetaAgentHarness(system)
        
        # Step 2: Use the LLM to generate a problem statement
        print("\nGenerating problem from goal...")
        generator = LLMBriefGenerator()
        problem = generator.generate_problem(user_goal)
        
        print(f"\nGenerated Problem:")
        print(f"  Description: {problem.get('description', 'None')}")
        print(f"  Constraints: {problem.get('constraints', {})}")
        
        # Step 3: Run adaptive discovery
        print("\n" + "-"*60)
        print("Starting Discovery Process")
        print("-"*60)
        
        results = await harness.run_adaptive_discovery(
            problems=[problem],
            max_generations=5,
            target_fitness=0.85,
            enable_pivoting=True
        )
        
        return results
        
    except AttributeError as e:
        print(f"\nError: {e}")
        print("Note: MetaAgentHarness may not have run_adaptive_discovery method")
        print("Using basic run instead...")
        
        # Fallback to basic generation run
        results = await system.run_generation(problem)
        return results
        
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main entry point"""
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\nWarning: OPENAI_API_KEY not set")
        print("The system will run but won't use LLM for problem generation")
        print("Set it with: export OPENAI_API_KEY=sk-your-key-here\n")
    
    # Default goal or get from command line
    if len(sys.argv) > 1:
        user_goal = " ".join(sys.argv[1:])
    else:
        user_goal = "Find the cheapest way to deploy ENTIENT with blockchain proof"
    
    print(f"\nLaunching ENTIENT with goal:")
    print(f'  "{user_goal}"')
    
    # Run the discovery
    results = asyncio.run(run_entient_with_prompt(user_goal))
    
    if results:
        print("\n" + "="*60)
        print("Discovery Complete")
        print("="*60)
        
        if isinstance(results, dict):
            for key, value in results.items():
                if key != 'metrics_history':  # Skip verbose history
                    print(f"  {key}: {value}")
        else:
            print(results)
    else:
        print("\nDiscovery failed or returned no results")

if __name__ == "__main__":
    main()