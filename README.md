# Mouse Behavior Labeling with Gemini

This script uses the Gemini API to label mouse behaviors in video files using human-annotated segments as reference.

## Usage

1. Install dependencies with [uv](https://github.com/astral-sh/uv):
   ```sh
   uv pip install -r requirements.txt
   ```
2. Edit `label_mouse_behavior_with_gemini.py` to set your Google Cloud project ID and the video base name you want to process.
3. Run the script:
   ```sh
   python label_mouse_behavior_with_gemini.py
   ```

Dependencies are listed in `requirements.txt` and should be installed using `uv`. 