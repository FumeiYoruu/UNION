"""
LLM-as-Judge Pipeline for Evaluating Long-Form Stories using LangChain
Uses binary yes/no questions for evaluation criteria
"""

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import csv
from pathlib import Path
import json


# Define the structure for LLM output
class StoryEvaluationOutput(BaseModel):
    """Schema for binary story evaluation output"""
    has_clear_plot_structure: bool = Field(description="Does the story have a clear beginning, middle, and end?")
    plot_events_logical: bool = Field(description="Do the plot events follow logically from one another?")
    repetitive_plot_structure: bool = Field(description="Does the story have a repetitive plot structure?")
    characters_well_defined: bool = Field(description="Are the main characters clearly defined with distinct personalities?")
    characters_show_growth: bool = Field(description="Do characters experience meaningful development or change?")
    pacing_appropriate: bool = Field(description="Is the pacing appropriate for the story type and length?")
    maintains_reader_interest: bool = Field(description="Does the story maintain reader interest throughout?")
    dialogue_sounds_natural: bool = Field(description="Does the dialogue sound natural and authentic?")
    dialogue_advances_story: bool = Field(description="Does dialogue serve to advance plot or reveal character?")
    has_unique_elements: bool = Field(description="Does the story contain original or unique elements?")
    avoids_cliches: bool = Field(description="Does the story avoid overused tropes and clichés?")
    evokes_emotional_response: bool = Field(description="Does the story successfully evoke an emotional response?")
    creates_immersion: bool = Field(description="Does the writing create an immersive experience?")
    prose_is_clear: bool = Field(description="Is the prose clear and easy to understand?")
    writing_has_voice: bool = Field(description="Does the writing demonstrate a distinct authorial voice?")
    grammar_correct: bool = Field(description="Is the grammar and syntax generally correct?")
    word_choice_effective: bool = Field(description="Are word choices effective and purposeful?")
    
    strengths: List[str] = Field(description="List of 2-4 specific strengths of the story")
    weaknesses: List[str] = Field(description="List of 2-4 specific weaknesses of the story")
    overall_recommendation: str = Field(description="One of: 'publish', 'revise', 'reject' (or 'good'/'bad' which will be mapped)")
    revision_priorities: List[str] = Field(description="Top 3 priorities for improvement if applicable")


@dataclass
class StoryEvaluation:
    """Stores evaluation results for a story"""
    story_id: str
    story_title: str
    
    # Plot criteria
    has_clear_plot_structure: bool
    plot_events_logical: bool
    repetitive_plot_structure: bool
    
    # Character criteria
    characters_well_defined: bool
    characters_show_growth: bool
    
    # Pacing criteria
    pacing_appropriate: bool
    maintains_reader_interest: bool
    
    # Dialogue criteria
    dialogue_sounds_natural: bool
    dialogue_advances_story: bool
    
    # Creativity criteria
    has_unique_elements: bool
    avoids_cliches: bool
    
    # Emotional impact criteria
    evokes_emotional_response: bool
    creates_immersion: bool
    
    # Writing style criteria
    prose_is_clear: bool
    writing_has_voice: bool
    grammar_correct: bool
    word_choice_effective: bool
    
    # Summary fields
    total_yes_count: int
    total_criteria: int
    pass_rate: float
    strengths: List[str]
    weaknesses: List[str]
    overall_recommendation: str
    revision_priorities: List[str]


class LLMJudgePipeline:
    """Pipeline for evaluating stories using LangChain and Claude"""
    
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        # Initialize LangChain LLM
        self.llm = ChatAnthropic(
            model=model,
            anthropic_api_key=api_key,
            temperature=0.3  # Lower temperature for more consistent judgments
        )
        
        # Setup output parser
        self.parser = JsonOutputParser(pydantic_object=StoryEvaluationOutput)
        
        # Define evaluation criteria categories
        self.criteria_categories = {
            "plot": ["has_clear_plot_structure", "plot_events_logical", "repetitive_plot_structure"],
            "character": ["characters_well_defined", "characters_show_growth"],
            "pacing": ["pacing_appropriate", "maintains_reader_interest"],
            "dialogue": ["dialogue_sounds_natural", "dialogue_advances_story"],
            "creativity": ["has_unique_elements", "avoids_cliches"],
            "emotional_impact": ["evokes_emotional_response", "creates_immersion"],
            "writing_style": ["prose_is_clear", "writing_has_voice", "grammar_correct", "word_choice_effective"]
        }
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert literary critic conducting a rigorous evaluation of creative writing. 
You will answer a series of YES/NO questions about the story to assess its quality across multiple dimensions.

Be objective and honest in your assessments. A "YES" means the criterion is clearly met; a "NO" means it is not met or only partially met."""),
            ("human", """Evaluate the following story by answering YES or NO to each criterion:

**Story Title:** {story_title}

**Story Text:**
{story_text}

---

**Evaluation Criteria (Answer YES or NO for each):**

PLOT:
1. Does the story have a clear beginning, middle, and end?
2. Do the plot events follow logically from one another?
3. Does the story have a repetitive plot structure?

CHARACTER:
4. Are the main characters clearly defined with distinct personalities?
5. Do characters experience meaningful development or change?

PACING:
6. Is the pacing appropriate for the story type and length?
7. Does the story maintain reader interest throughout?

DIALOGUE:
8. Does the dialogue sound natural and authentic?
9. Does dialogue serve to advance plot or reveal character?

CREATIVITY:
10. Does the story contain original or unique elements?
11. Does the story avoid overused tropes and clichés?

EMOTIONAL IMPACT:
12. Does the story successfully evoke an emotional response?
13. Does the writing create an immersive experience?

WRITING STYLE:
14. Is the prose clear and easy to understand?
15. Does the writing demonstrate a distinct authorial voice?
16. Is the grammar and syntax generally correct?
17. Are word choices effective and purposeful?

Additionally, provide:
- 2-4 specific strengths
- 2-4 specific weaknesses  
- Overall binary rating: 'good', 'bad'
- Top 3 priorities for improvement (if applicable)

{format_instructions}""")
        ])
        
        # Create the chain
        self.chain = self.prompt | self.llm | self.parser

    def _normalize_result(self, result) -> Dict[str, object]:
        """Normalize parser/LLM output into a plain dict with expected keys.

        The LLM/parser may return a Pydantic model, a dict-like object, or
        another structure. This helper attempts to coerce into a dict and
        normalize a couple of legacy key names (e.g. 'overall_binary_rating').
        """
        # If it's a pydantic/BaseModel-like, try .dict()
        try:
            if hasattr(result, "dict"):
                data = result.dict()
            else:
                # Try mapping or attribute-based access
                data = dict(result)
        except Exception:
            # Fallback: try __dict__
            try:
                data = vars(result)
            except Exception:
                raise ValueError("Unable to normalize LLM result to dict")

        # Normalize key names
        if "overall_binary_rating" in data and "overall_recommendation" not in data:
            data["overall_recommendation"] = data.pop("overall_binary_rating")

        # Map simple binary terms to our recommendation categories if needed
        if "overall_recommendation" in data and isinstance(data["overall_recommendation"], str):
            val = data["overall_recommendation"].strip().lower()
            map_good_bad = {"good": "publish", "bad": "reject"}
            data["overall_recommendation"] = map_good_bad.get(val, val)

        # Ensure list fields exist
        for list_field in ("strengths", "weaknesses", "revision_priorities"):
            if list_field not in data or data[list_field] is None:
                data[list_field] = []

        return data
    
    def evaluate_story(
        self, 
        story: str, 
        story_id: str = "story_1",
        story_title: str = "Untitled"
    ) -> StoryEvaluation:
        """Evaluates a single story using the LLM judge"""
        
        print(f"Evaluating: {story_title} (ID: {story_id})")
        
        # Invoke the chain
        result = self.chain.invoke({
            "story_title": story_title,
            "story_text": story,
            "format_instructions": self.parser.get_format_instructions()
        })

        # Normalize parser output to plain dict
        res = self._normalize_result(result)

        # Count yes responses: include the repetitive plot criterion
        criteria_fields = [
            "has_clear_plot_structure", "plot_events_logical", "repetitive_plot_structure",
            "characters_well_defined", "characters_show_growth",
            "pacing_appropriate", "maintains_reader_interest",
            "dialogue_sounds_natural", "dialogue_advances_story",
            "has_unique_elements", "avoids_cliches",
            "evokes_emotional_response", "creates_immersion",
            "prose_is_clear", "writing_has_voice",
            "grammar_correct", "word_choice_effective"
        ]

        def _is_yes(v) -> bool:
            if isinstance(v, bool):
                return v
            if v is None:
                return False
            return str(v).strip().lower() in ("yes", "y", "true", "1")

        yes_count = sum(1 for field in criteria_fields if _is_yes(res.get(field)))
        total_criteria = len(criteria_fields)
        pass_rate = (yes_count / total_criteria) * 100 if total_criteria else 0.0
        
        # Create evaluation object
        evaluation = StoryEvaluation(
            story_id=story_id,
            story_title=story_title,
            has_clear_plot_structure=res.get("has_clear_plot_structure", False),
            plot_events_logical=res.get("plot_events_logical", False),
            repetitive_plot_structure=res.get("repetitive_plot_structure", False),
            characters_well_defined=res.get("characters_well_defined", False),
            characters_show_growth=res.get("characters_show_growth", False),
            pacing_appropriate=res.get("pacing_appropriate", False),
            maintains_reader_interest=res.get("maintains_reader_interest", False),
            dialogue_sounds_natural=res.get("dialogue_sounds_natural", False),
            dialogue_advances_story=res.get("dialogue_advances_story", False),
            has_unique_elements=res.get("has_unique_elements", False),
            avoids_cliches=res.get("avoids_cliches", False),
            evokes_emotional_response=res.get("evokes_emotional_response", False),
            creates_immersion=res.get("creates_immersion", False),
            prose_is_clear=res.get("prose_is_clear", False),
            writing_has_voice=res.get("writing_has_voice", False),
            grammar_correct=res.get("grammar_correct", False),
            word_choice_effective=res.get("word_choice_effective", False),
            total_yes_count=yes_count,
            total_criteria=total_criteria,
            pass_rate=pass_rate,
            strengths=res.get("strengths", []),
            weaknesses=res.get("weaknesses", []),
            overall_recommendation=res.get("overall_recommendation", "revise"),
            revision_priorities=res.get("revision_priorities", [])
        )
        
        print(f"✓ Completed. Pass Rate: {pass_rate:.1f}% ({yes_count}/{total_criteria})")
        print(f"  Recommendation: {evaluation.overall_recommendation.upper()}\n")
        
        return evaluation
    
    def evaluate_batch(
        self, 
        stories: List[Dict[str, str]]
    ) -> List[StoryEvaluation]:
        """
        Evaluates multiple stories in batch
        
        Args:
            stories: List of dicts with keys 'id', 'title', 'text'
        
        Returns:
            List of StoryEvaluation objects
        """
        evaluations = []
        
        for i, story_data in enumerate(stories, 1):
            print(f"[{i}/{len(stories)}] ", end="")
            
            evaluation = self.evaluate_story(
                story=story_data["text"],
                story_id=story_data.get("id", f"story_{i}"),
                story_title=story_data.get("title", "Untitled")
            )
            evaluations.append(evaluation)
        
        return evaluations
    
    def save_results(
        self, 
        evaluations: List[StoryEvaluation], 
        output_path: str = "story_evaluations.csv"
    ):
        """Saves evaluation results to CSV"""
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                "Story ID", "Title", "Pass Rate %", "Yes Count", "Total Criteria",
                "Clear Plot", "Logical Events", "Repetitive Plot Structure", "Defined Characters", "Character Growth",
                "Good Pacing", "Maintains Interest", "Natural Dialogue", "Dialogue Advances Story",
                "Unique Elements", "Avoids Cliches", "Emotional Response", "Creates Immersion",
                "Clear Prose", "Has Voice", "Correct Grammar", "Effective Words",
                "Recommendation", "Strengths", "Weaknesses", "Revision Priorities"
            ])
            
            # Data
            for eval in evaluations:
                writer.writerow([
                    eval.story_id,
                    eval.story_title,
                    f"{eval.pass_rate:.1f}",
                    eval.total_yes_count,
                    eval.total_criteria,
                    "YES" if eval.has_clear_plot_structure else "NO",
                    "YES" if eval.plot_events_logical else "NO",
                    "YES" if eval.repetitive_plot_structure else "NO",
                    "YES" if eval.characters_well_defined else "NO",
                    "YES" if eval.characters_show_growth else "NO",
                    "YES" if eval.pacing_appropriate else "NO",
                    "YES" if eval.maintains_reader_interest else "NO",
                    "YES" if eval.dialogue_sounds_natural else "NO",
                    "YES" if eval.dialogue_advances_story else "NO",
                    "YES" if eval.has_unique_elements else "NO",
                    "YES" if eval.avoids_cliches else "NO",
                    "YES" if eval.evokes_emotional_response else "NO",
                    "YES" if eval.creates_immersion else "NO",
                    "YES" if eval.prose_is_clear else "NO",
                    "YES" if eval.writing_has_voice else "NO",
                    "YES" if eval.grammar_correct else "NO",
                    "YES" if eval.word_choice_effective else "NO",
                    eval.overall_recommendation.upper(),
                    " | ".join(eval.strengths),
                    " | ".join(eval.weaknesses),
                    " | ".join(eval.revision_priorities)
                ])
        
        print(f"✓ Results saved to {output_path}")
    
    def save_detailed_report(
        self, 
        evaluations: List[StoryEvaluation],
        output_path: str = "detailed_report.txt"
    ):
        """Saves a detailed human-readable report"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("STORY EVALUATION REPORT - BINARY CRITERIA\n")
            f.write("=" * 80 + "\n\n")
            
            for eval in evaluations:
                f.write(f"\n{'=' * 80}\n")
                f.write(f"Story: {eval.story_title} (ID: {eval.story_id})\n")
                f.write(f"{'=' * 80}\n\n")
                
                f.write(f"OVERALL PASS RATE: {eval.pass_rate:.1f}% ({eval.total_yes_count}/{eval.total_criteria})\n")
                f.write(f"RECOMMENDATION: {eval.overall_recommendation.upper()}\n\n")
                
                f.write("CRITERIA EVALUATION:\n\n")
                
                f.write("PLOT:\n")
                f.write(f"  Clear structure (beginning/middle/end): {'✓ YES' if eval.has_clear_plot_structure else '✗ NO'}\n")
                f.write(f"  Logical event progression:              {'✓ YES' if eval.plot_events_logical else '✗ NO'}\n\n")
                f.write(f"  Repetitive plot structure:               {'✓ YES' if eval.repetitive_plot_structure else '✗ NO'}\n\n")
                
                f.write("CHARACTER:\n")
                f.write(f"  Well-defined characters:                {'✓ YES' if eval.characters_well_defined else '✗ NO'}\n")
                f.write(f"  Character growth/development:           {'✓ YES' if eval.characters_show_growth else '✗ NO'}\n\n")
                
                f.write("PACING:\n")
                f.write(f"  Appropriate pacing:                     {'✓ YES' if eval.pacing_appropriate else '✗ NO'}\n")
                f.write(f"  Maintains reader interest:              {'✓ YES' if eval.maintains_reader_interest else '✗ NO'}\n\n")
                
                f.write("DIALOGUE:\n")
                f.write(f"  Natural and authentic:                  {'✓ YES' if eval.dialogue_sounds_natural else '✗ NO'}\n")
                f.write(f"  Advances plot/reveals character:        {'✓ YES' if eval.dialogue_advances_story else '✗ NO'}\n\n")
                
                f.write("CREATIVITY:\n")
                f.write(f"  Original/unique elements:               {'✓ YES' if eval.has_unique_elements else '✗ NO'}\n")
                f.write(f"  Avoids clichés:                         {'✓ YES' if eval.avoids_cliches else '✗ NO'}\n\n")
                
                f.write("EMOTIONAL IMPACT:\n")
                f.write(f"  Evokes emotional response:              {'✓ YES' if eval.evokes_emotional_response else '✗ NO'}\n")
                f.write(f"  Creates immersion:                      {'✓ YES' if eval.creates_immersion else '✗ NO'}\n\n")
                
                f.write("WRITING STYLE:\n")
                f.write(f"  Clear prose:                            {'✓ YES' if eval.prose_is_clear else '✗ NO'}\n")
                f.write(f"  Distinct voice:                         {'✓ YES' if eval.writing_has_voice else '✗ NO'}\n")
                f.write(f"  Correct grammar:                        {'✓ YES' if eval.grammar_correct else '✗ NO'}\n")
                f.write(f"  Effective word choice:                  {'✓ YES' if eval.word_choice_effective else '✗ NO'}\n\n")
                
                f.write("STRENGTHS:\n")
                for strength in eval.strengths:
                    f.write(f"  • {strength}\n")
                f.write("\n")
                
                f.write("WEAKNESSES:\n")
                for weakness in eval.weaknesses:
                    f.write(f"  • {weakness}\n")
                f.write("\n")
                
                if eval.revision_priorities:
                    f.write("REVISION PRIORITIES:\n")
                    for i, priority in enumerate(eval.revision_priorities, 1):
                        f.write(f"  {i}. {priority}\n")
                    f.write("\n")
        
        print(f"✓ Detailed report saved to {output_path}")
    
    def print_summary(self, evaluations: List[StoryEvaluation]):
        """Prints a summary of evaluations"""
        
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        
        n = len(evaluations)
        avg_pass_rate = (sum(e.pass_rate for e in evaluations) / n) if n else 0.0
        
        # Count recommendations
        publish_count = sum(1 for e in evaluations if e.overall_recommendation == "publish")
        revise_count = sum(1 for e in evaluations if e.overall_recommendation == "revise")
        reject_count = sum(1 for e in evaluations if e.overall_recommendation == "reject")
        
        print(f"\nTotal Stories Evaluated: {n}")
        print(f"Average Pass Rate: {avg_pass_rate:.1f}%\n")
        
        print("Recommendations:")
        if n:
            print(f"  Publish: {publish_count} ({publish_count/n*100:.1f}%)")
            print(f"  Revise:  {revise_count} ({revise_count/n*100:.1f}%)")
            print(f"  Reject:  {reject_count} ({reject_count/n*100:.1f}%)\n")
        else:
            print(f"  Publish: {publish_count}")
            print(f"  Revise:  {revise_count}")
            print(f"  Reject:  {reject_count}\n")
        
        # Calculate pass rate by category
        print("Pass Rates by Category:")
        
        categories = {
            "Plot": ["has_clear_plot_structure", "plot_events_logical", "repetitive_plot_structure"],
            "Character": ["characters_well_defined", "characters_show_growth"],
            "Pacing": ["pacing_appropriate", "maintains_reader_interest"],
            "Dialogue": ["dialogue_sounds_natural", "dialogue_advances_story"],
            "Creativity": ["has_unique_elements", "avoids_cliches"],
            "Emotional Impact": ["evokes_emotional_response", "creates_immersion"],
            "Writing Style": ["prose_is_clear", "writing_has_voice", "grammar_correct", "word_choice_effective"]
        }
        
        for category, criteria in categories.items():
            total_yes = sum(
                sum(1 for c in criteria if getattr(e, c))
                for e in evaluations
            )
            total_possible = len(criteria) * n
            category_rate = (total_yes / total_possible) * 100
            print(f"  {category}: {category_rate:.1f}%")
        
        print("=" * 80 + "\n")


# Example usage
if __name__ == "__main__":
    # Initialize pipeline (you'll need to set your API key)
    pipeline = LLMJudgePipeline(api_key="your-api-key-here")
    
    # Example stories to evaluate
    stories = [
        {
            "id": "story_001",
            "title": "The Last Lighthouse",
            "text": """The lighthouse keeper had been alone for three years when the storm came. 
            
Not just any storm—the kind that meteorologists on the mainland called "once in a century," the kind that made the sea foam white as far as the eye could see. Marcus stood at the top of the lighthouse, watching the waves climb higher and higher, each one threatening to swallow the rocky outcrop that had been his home since the accident.

He thought about Sarah then, as he always did during storms. She would have loved this—the raw power of nature, the way the wind howled like a living thing. But Sarah was gone, and he was here, and the light had to keep turning.

The wave that finally broke over the lighthouse was taller than the structure itself. Marcus held tight to the railing as water crashed through the windows, as the building shook on its foundations. And in that moment, suspended between sea and sky, he saw her—Sarah, standing on the water, smiling that knowing smile.

"It's time," she said, though her voice was just the wind.

The lighthouse fell. But Marcus, finally, was free."""
        },
        {
            "id": "story_002", 
            "title": "Coffee at Midnight",
            "text": """She always ordered the same thing: black coffee, no sugar, extra hot.

I worked the graveyard shift at the diner for two years before I got up the courage to ask her name. "June," she said, and nothing more. 

June came every Tuesday at midnight, sat in the same booth by the window, and wrote in a leather journal until 2 AM. I'd refill her coffee three times. We never spoke beyond pleasantries.

Then one Tuesday, she didn't come. Or the next. On the third Tuesday, I found an envelope taped to her booth. Inside, a note: "Thank you for the coffee. Thank you for not asking. - J"

And a manuscript. Her manuscript. Dedicated to "the silent witness who kept me company through the darkest chapters."

I still work that shift. Still make her coffee at midnight, just in case."""
        }
    ]
    
    # Evaluate stories
    print("Starting LangChain evaluation pipeline...\n")
    evaluations = pipeline.evaluate_batch(stories)
    
    # Save results
    pipeline.save_results(evaluations, "story_evaluations.csv")
    pipeline.save_detailed_report(evaluations, "detailed_report.txt")
    
    # Print summary
    pipeline.print_summary(evaluations)