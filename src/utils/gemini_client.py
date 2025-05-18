from google import genai
from google.genai.types import GenerateContentConfig, Part

# Gemini config
def get_config():
    return GenerateContentConfig(
        temperature=0,
        top_p=1,
        max_output_tokens=8000,
        response_mime_type="application/json",
    )

# Prompt for Gemini
mouse_behavior_instructions = """
You are a Mouse Behavior Labeler.
Your task is to analyze a video of mice and segment it into periods of distinct behaviors.
Label each segment with one of the following behaviors: attack, investigation, mount, other.
Use the video and the provided annotation JSON as references.
For each segment, provide:
- start and end time (HRS:MIN:SEC)
- behavior label
- a brief description of the behavior
"""

def get_gemini_response(video_data, annotation_summary, start_segment, api_key, model_id):
    """Get response from Gemini model"""
    # Initialize client with provided API key
    client = genai.Client(api_key=api_key)
    
    return client.models.generate_content(
        model=model_id,
        contents=[
            Part.from_bytes(data=video_data, mime_type="video/mp4"),
            Part.from_text(text=mouse_behavior_instructions),
            f"Here are the first {start_segment} human-labeled segments for reference:",
            Part.from_text(text=annotation_summary),
        ],
        config=get_config(),
    ) 