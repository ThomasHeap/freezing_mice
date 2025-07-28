from openai import OpenAI
from ..config.prompts import PROMPTS
from ..models.mouse_behavior import MouseBehaviorSegment, ScratchAidSegment, GroomingSegment, MouseBoxSegment, FreezingSegment, ForagingSegment, MouseVentralSegment
from .gcs_utils import get_signed_url, download_gcs_file, cleanup_temp_file
import json
import os

def get_qwen_response(video_path: str, annotation_summary: str, start_segment: int, 
                     prompt_template: str = "default", full_example: str = None, 
                     full_example_video: str = None, example_clips: dict = None):
    """
    Get response from Qwen model
    
    Args:
        video_path: Path to the video file
        annotation_summary: JSON string of annotation examples
        start_segment: Number of segments used as context
        prompt_template: Name of the prompt template to use
        full_example: Optional fully annotated example JSON
        full_example_video: Optional video path for the fully annotated example
        example_clips: Optional dictionary of example clips by behavior type
    """
    try:
        # Initialize Qwen client
        print(f"Initializing Qwen client...")
        # Set the API key for OpenAI client
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY environment variable not set")
        
        # Set the environment variable that OpenAI SDK expects
        os.environ["OPENAI_API_KEY"] = api_key
        
        client = OpenAI(
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
        
        # Get the appropriate prompt template
        print(f"Using prompt template: {prompt_template}")
        prompt_text = PROMPTS[prompt_template]
        
        messages = []
        # Add a system prompt as the first message
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": "You are a mouse behavior annotator. Make sure to annotate the full length of the video."}]
        })
        
        # Build user content
        user_content = []
        # Add the main prompt
        user_content.append({
            "type": "text",
            "text": prompt_text
        })
        
        # Add examples of the behavior if provided
        if example_clips:
            example_text = "Here are example clips for each behavior:\n"
            for behavior, clips in example_clips.items():
                example_text += f"Behavior: {behavior}\n"
                for clip in clips:
                    example_text += f"- {clip}\n"
            user_content.append({
                "type": "text",
                "text": example_text
            })
            # Optionally, add example videos as video_url
            for behavior, clips in example_clips.items():
                for clip in clips:
                    # Handle GCS URLs for example clips
                    if clip.startswith("gs://"):
                        clip_url = get_signed_url(clip)
                        if clip_url is None:
                            print(f"Warning: Failed to generate signed URL for example clip {clip}")
                            continue
                    else:
                        clip_url = clip if clip.startswith("http") or clip.startswith("file://") else f"file://{clip}"
                    
                    user_content.append({
                        "type": "video_url",
                        "video_url": {"url": clip_url}
                    })

        # Add full example video and annotations if provided
        if full_example and full_example_video:
            # Handle GCS URLs for full example video
            if full_example_video.startswith("gs://"):
                example_video_url = get_signed_url(full_example_video)
                if example_video_url is None:
                    raise ValueError(f"Failed to generate signed URL for example video {full_example_video}")
            else:
                example_video_url = full_example_video if full_example_video.startswith("http") or full_example_video.startswith("file://") else f"file://{full_example_video}"
            
            user_content.extend([
                {
                    "type": "text",
                    "text": "Here is an example video and its annotations:"
                },
                {
                    "type": "video_url",
                    "video_url": {"url": example_video_url}
                },
                {
                    "type": "text",
                    "text": f"Annotations: {full_example}"
                }
            ])
            
        # Add the current video's segments
        if start_segment > 0 and annotation_summary is not None:
            if full_example and full_example_video:
                user_content.append({
                    "type": "text",
                    "text": "\nHere are the first few segments from the current video for reference:"
                })
            else:
                user_content.append({
                    "type": "text",
                    "text": f"Here are the first {start_segment} human-labeled segments for reference:"
                })
            user_content.append({
                "type": "text",
                "text": annotation_summary
            })
            
        # Handle GCS URLs - use signed URLs for private buckets
        if video_path.startswith("gs://"):
            # Generate signed URL for GCS object
            video_url = get_signed_url(video_path)
            if video_url is None:
                raise ValueError(f"Failed to generate signed URL for {video_path}")
        else:
            # For local files or HTTP URLs
            video_url = video_path if video_path.startswith("http") or video_path.startswith("file://") else f"file://{video_path}"
        
        # Add the main video (as video_url)
        user_content.extend([
            {
                "type": "text",
                "text": "Here is the video to be annotated, only annotate this video, do not annotate other videos:"
            },
            {
                "type": "video_url",
                "video_url": {"url": video_url}
            }
        ])
        
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        print("Sending request to Qwen API...")
        
        # Use qwen-vl-max model for video analysis
        response = client.chat.completions.create(
            model="qwen-vl-max",
            messages=messages,
            temperature=0,
            max_tokens=8192,  # Qwen's maximum allowed value
            response_format={ "type": "json_object" }
        )
        
        if response is None:
            print("Error: Qwen API returned None response")
            return None
            
        print("Successfully received response from Qwen API")
        return response
        
    except Exception as e:
        print(f"Error in get_qwen_response: {str(e)}")
        return None 