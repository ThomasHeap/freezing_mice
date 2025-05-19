from google import genai
from google.genai.types import GenerateContentConfig, Part
from ..config.prompts import PROMPTS
from ..models.mouse_behavior import MouseBehaviorSegment, ScratchAidSegment
import json

# Gemini config
def get_config(schema):
    return GenerateContentConfig(
        temperature=0,
        top_p=1,
        max_output_tokens=40000,
        response_mime_type="application/json",
        response_schema=schema,
    )



def get_gemini_response(video_data, annotation_summary, start_segment, project, location, model_id, prompt_template="default", full_example=None, full_example_video=None):
    """
    Get response from Gemini model
    
    Args:
        video_data: Binary video data
        annotation_summary: JSON string of annotation examples
        start_segment: Number of segments used as context
        project: Google Cloud project ID
        location: Google Cloud location
        model_id: Gemini model ID
        prompt_template: Name of the prompt template to use (default: "default")
        full_example: Optional fully annotated example JSON
        full_example_video: Optional video data for the fully annotated example
    """
    try:
        # Initialize client with provided API key
        print(f"Initializing Gemini client with model: {model_id}")
        client = genai.Client(vertexai=True, project=project, location=location)
        
        # Get the appropriate prompt template
        print(f"Using prompt template: {prompt_template}")
        
        # Prepare the content parts
        content_parts = []
        
        # Add full example video and annotations if provided
        if full_example and full_example_video:
            content_parts.extend([
                "Here is an example video:",
                Part.from_bytes(data=full_example_video, mime_type="video/mp4"),
                "And its annotations:",
                Part.from_text(text=json.dumps(full_example, indent=2)),
                "\nNow, here is the video you need to analyze:",
            ])
        
        # Add the main video and its annotations
        content_parts.extend([
            Part.from_bytes(data=video_data, mime_type="video/mp4"),
            Part.from_text(text=PROMPTS[prompt_template]),
        ])
        
        # Add the current video's segments
        if start_segment > 0:
            if full_example and full_example_video:
                content_parts.append("\nHere are the first few segments from the current video for reference:")
            else:
                content_parts.append(f"Here are the first {start_segment} human-labeled segments for reference:")
            
            content_parts.append(Part.from_text(text=annotation_summary))
    
        
        print("Sending request to Gemini API...")
        
        if prompt_template == "scratch_aid":
            schema = list[ScratchAidSegment]
        else:
            schema = list[MouseBehaviorSegment]
            
        response = client.models.generate_content(
            model=model_id,
            contents=content_parts,
            config=get_config(schema),
        )
        
        if response is None:
            print("Error: Gemini API returned None response")
            return None
            
        print("Successfully received response from Gemini API")
        return response
        
    except Exception as e:
        print(f"Error in get_gemini_response: {str(e)}")
        return None 