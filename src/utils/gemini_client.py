from google import genai
from google.genai.types import GenerateContentConfig, Part
from ..config.prompts import PROMPTS
from ..models.mouse_behavior import MouseBehaviorSegment, ScratchAidSegment, GroomingSegment, MouseBoxSegment, FreezingSegment, ForagingSegment, MouseVentralSegment
import json

# Gemini config
def get_config(schema):
    return GenerateContentConfig(
        temperature=0,
        top_p=1,
        max_output_tokens=65535,
        response_mime_type="application/json",
        response_schema=schema,
    )



def get_gemini_response(video_uri, annotation_summary, start_segment, project, location, model_id, prompt_template="default", 
                        full_example=None, full_example_video=None, example_clips=None):
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
        content_parts = [Part.from_text(text=PROMPTS[prompt_template]),]
            
        # Add examples of the behavior if provided
        if example_clips:
            content_parts.append("Here are example clips for each behavior:")
            for behavior, clips in example_clips.items():
                content_parts.append(f"Behavior: {behavior}")
                for clip in clips:
                    print(clip)
                    content_parts.extend([
                        Part.from_uri(file_uri=clip, mime_type="video/mp4"),
                    ])

        # Add full example video and annotations if provided
        if full_example and full_example_video:
            content_parts.extend([
                "Here is an example video:",
                Part.from_uri(file_uri=full_example_video, mime_type="video/mp4"),
                "and its annotations:",
                Part.from_text(text=json.dumps(full_example, indent=2)),
                
            ])
            
        # Add the current video's segments
        if start_segment > 0 and annotation_summary is not None:
            if full_example and full_example_video:
                content_parts.append("\nHere are the first few segments from the current video for reference:")
            else:
                content_parts.append(f"Here are the first {start_segment} human-labeled segments for reference:")
            
            content_parts.append(Part.from_text(text=annotation_summary))
            
        # Add the main video and its annotations
        content_parts.extend([
            "Here is the video to be annotated, only annotate this video, do not annotate other videos:",
            Part.from_uri(file_uri=video_uri, mime_type="video/mp4"),
        ])
        

    
        
        print("Sending request to Gemini API...")
        
        if prompt_template == "scratch_aid":
            schema = list[ScratchAidSegment]
        elif prompt_template == "grooming":
            schema = list[GroomingSegment]
        elif prompt_template == "calms":
            schema = list[MouseBehaviorSegment]
        elif prompt_template == "mouse_box":
            schema = list[MouseBoxSegment]
        elif prompt_template == "freezing":
            schema = list[FreezingSegment]
        elif prompt_template == "foraging":
            schema = list[ForagingSegment]
        elif prompt_template == "mouse_ventral1" or prompt_template == "mouse_ventral2":
            schema = list[MouseVentralSegment]
        else:
            raise ValueError(f"Invalid prompt template: {prompt_template}")
            
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