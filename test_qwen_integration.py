#!/usr/bin/env python3
"""
Test script for Qwen integration with mouse behavior labeling
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.qwen_client import get_qwen_response

def test_qwen_integration():
    """Test the Qwen integration with a sample video"""
    
    # Check if API key is set
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("Error: DASHSCOPE_API_KEY environment variable not set")
        print("Please set your DashScope API key:")
        print("export DASHSCOPE_API_KEY='your-api-key-here'")
        return False
    
    # Test parameters
    video_path = "test_video.mp4"  # You'll need to provide a test video
    annotation_summary = None
    start_segment = 0
    prompt_template = "calms"
    
    print("Testing Qwen integration...")
    print(f"Video path: {video_path}")
    print(f"Prompt template: {prompt_template}")
    
    # Check if test video exists
    if not os.path.exists(video_path):
        print(f"Warning: Test video {video_path} not found")
        print("Please provide a test video file to run this test")
        return False
    
    try:
        # Call Qwen API
        response = get_qwen_response(
            video_path=video_path,
            annotation_summary=annotation_summary,
            start_segment=start_segment,
            prompt_template=prompt_template
        )
        
        if response is None:
            print("❌ Test failed: No response from Qwen API")
            return False
        
        print("✅ Test successful: Received response from Qwen API")
        print(f"Response content: {response.choices[0].message.content[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_qwen_integration()
    sys.exit(0 if success else 1) 