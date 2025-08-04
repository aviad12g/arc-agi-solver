"""Prompt templates for LLM-based DSL program generation.

This module provides templates for creating prompts that guide the LLM
to generate valid DSL programs for ARC puzzle transformations.
"""

import json
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """Template for generating LLM prompts."""
    
    system_prompt: str
    user_template: str
    examples: List[Dict[str, Any]]
    
    def format_prompt(self, task_description: Dict[str, Any]) -> str:
        """Format the complete prompt with task description.
        
        Args:
            task_description: Structured description of the transformation task
            
        Returns:
            Formatted prompt string
        """
        # Format examples
        examples_text = ""
        for i, example in enumerate(self.examples, 1):
            examples_text += f"\nExample {i}:\n"
            examples_text += f"Input: {example['input']}\n"
            examples_text += f"Output: {example['output']}\n"
            examples_text += f"Program: {example['program']}\n"
        
        # Format task description
        task_json = json.dumps(task_description, indent=2)
        
        # Combine all parts
        full_prompt = f"{self.system_prompt}\n\n"
        full_prompt += f"Examples:{examples_text}\n\n"
        full_prompt += self.user_template.format(task_description=task_json)
        
        return full_prompt


def create_arc_prompt_template() -> PromptTemplate:
    """Create the standard ARC prompt template for DSL program generation.
    
    Returns:
        PromptTemplate configured for ARC tasks
    """
    system_prompt = """You are an expert at analyzing visual patterns and generating transformation programs for Abstract Reasoning Corpus (ARC) puzzles.

Your task is to analyze the input and target grids, then generate a sequence of operations that transforms the input into the target.

Available Operations:
- Rotate90: Rotate grid 90 degrees clockwise
- Rotate180: Rotate grid 180 degrees
- ReflectH: Reflect grid horizontally (left-right flip)
- ReflectV: Reflect grid vertically (up-down flip)
- Crop(r1, r2, c1, c2): Crop grid to region from (r1,c1) to (r2,c2)
- Paint(x, y, c): Paint pixel at position (x,y) with color c
- MapColors(mapping): Remap colors according to mapping dictionary
- PaintIf(predicate, action): Conditionally paint based on blob properties

Program Format:
- Operations are connected with " -> "
- Maximum program length is 4 operations
- Example: "Rotate90 -> ReflectH -> Paint(1, 2, 5)"

Focus on:
1. Geometric transformations (rotations, reflections)
2. Color changes and mappings
3. Spatial relationships between objects
4. Pattern completion and symmetry"""

    user_template = """Analyze this transformation task:

{task_description}

Based on the input and target grids, their blob properties, and transformation hints, generate a DSL program that transforms the input into the target.

Consider:
- Shape and size changes
- Color transformations
- Geometric operations (rotations, reflections)
- Spatial relationships
- Pattern symmetries

Program:"""

    # Example transformations for few-shot learning
    examples = [
        {
            "input": "2x2 grid with colors [1,2,3,4]",
            "output": "2x2 grid with colors [3,1,4,2] (90-degree rotation)",
            "program": "Rotate90"
        },
        {
            "input": "3x3 grid with blue square in center",
            "output": "3x3 grid with blue square reflected horizontally",
            "program": "ReflectH"
        },
        {
            "input": "4x4 grid with red and blue objects",
            "output": "4x4 grid with colors swapped (red->blue, blue->red)",
            "program": "MapColors({1: 2, 2: 1})"
        },
        {
            "input": "5x5 grid with pattern in top-left quadrant",
            "output": "2x2 grid containing only the pattern",
            "program": "Crop(0, 2, 0, 2)"
        }
    ]
    
    return PromptTemplate(
        system_prompt=system_prompt,
        user_template=user_template,
        examples=examples
    )


def create_few_shot_prompt_template(training_examples: List[Dict[str, Any]]) -> PromptTemplate:
    """Create a prompt template with custom few-shot examples.
    
    Args:
        training_examples: List of training examples with input/output/program
        
    Returns:
        PromptTemplate with custom examples
    """
    system_prompt = """You are an expert at analyzing visual patterns and generating transformation programs for Abstract Reasoning Corpus (ARC) puzzles.

Your task is to analyze the input and target grids, then generate a sequence of operations that transforms the input into the target.

Available Operations:
- Rotate90: Rotate grid 90 degrees clockwise
- Rotate180: Rotate grid 180 degrees
- ReflectH: Reflect grid horizontally (left-right flip)
- ReflectV: Reflect grid vertically (up-down flip)
- Crop(r1, r2, c1, c2): Crop grid to region from (r1,c1) to (r2,c2)
- Paint(x, y, c): Paint pixel at position (x,y) with color c
- MapColors(mapping): Remap colors according to mapping dictionary
- PaintIf(predicate, action): Conditionally paint based on blob properties

Program Format:
- Operations are connected with " -> "
- Maximum program length is 4 operations
- Example: "Rotate90 -> ReflectH -> Paint(1, 2, 5)"

Focus on identifying the core transformation pattern."""

    user_template = """Analyze this transformation task:

{task_description}

Generate a DSL program that transforms the input into the target.

Program:"""

    return PromptTemplate(
        system_prompt=system_prompt,
        user_template=user_template,
        examples=training_examples
    )


def create_chain_of_thought_prompt_template() -> PromptTemplate:
    """Create a prompt template that encourages chain-of-thought reasoning.
    
    Returns:
        PromptTemplate with chain-of-thought structure
    """
    system_prompt = """You are an expert at analyzing visual patterns and generating transformation programs for Abstract Reasoning Corpus (ARC) puzzles.

Your task is to analyze the input and target grids step by step, then generate a sequence of operations that transforms the input into the target.

Available Operations:
- Rotate90: Rotate grid 90 degrees clockwise
- Rotate180: Rotate grid 180 degrees
- ReflectH: Reflect grid horizontally (left-right flip)
- ReflectV: Reflect grid vertically (up-down flip)
- Crop(r1, r2, c1, c2): Crop grid to region from (r1,c1) to (r2,c2)
- Paint(x, y, c): Paint pixel at position (x,y) with color c
- MapColors(mapping): Remap colors according to mapping dictionary
- PaintIf(predicate, action): Conditionally paint based on blob properties

Think step by step:
1. Analyze the input grid structure
2. Analyze the target grid structure
3. Identify the key differences
4. Determine the transformation type
5. Generate the program"""

    user_template = """Analyze this transformation task:

{task_description}

Let me think step by step:

1. Input Analysis:
   - What patterns do I see in the input?
   - What are the key objects and their properties?

2. Target Analysis:
   - What patterns do I see in the target?
   - How do they relate to the input patterns?

3. Transformation Analysis:
   - What changed between input and target?
   - Is this a geometric transformation?
   - Is this a color transformation?
   - Is this a spatial transformation?

4. Program Generation:
   Based on my analysis, the transformation program is:

Program:"""

    examples = [
        {
            "input": "Input grid analysis shows a 3x3 pattern",
            "output": "Target shows the same pattern rotated 90 degrees",
            "program": "Rotate90"
        }
    ]
    
    return PromptTemplate(
        system_prompt=system_prompt,
        user_template=user_template,
        examples=examples
    )


def create_minimal_prompt_template() -> PromptTemplate:
    """Create a minimal prompt template for fast generation.
    
    Returns:
        Minimal PromptTemplate
    """
    system_prompt = """Generate DSL program to transform input grid to target grid.

Operations: Rotate90, Rotate180, ReflectH, ReflectV, Crop(r1,r2,c1,c2), Paint(x,y,c), MapColors(mapping)
Format: operation1 -> operation2 -> operation3
Max length: 4 operations"""

    user_template = """Input: {task_description}

Program:"""

    return PromptTemplate(
        system_prompt=system_prompt,
        user_template=user_template,
        examples=[]
    )