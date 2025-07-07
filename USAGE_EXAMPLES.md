# Mouse Behavior Labeling Script Usage Examples

This script has been updated to process folders of videos with timeout and retry functionality.

## Basic Usage

### Process a single video file:
```bash
python label_mouse_behavior_with_gemini.py \
    --input /path/to/video.mp4 \
    --project your-gcp-project \
    --prompt-template grooming \
    --output-dir ./results
```

### Process a local folder of videos:
```bash
python label_mouse_behavior_with_gemini.py \
    --input /path/to/video/folder \
    --project your-gcp-project \
    --prompt-template grooming \
    --output-dir ./results
```

### Process videos from Google Cloud Storage:
```bash
python label_mouse_behavior_with_gemini.py \
    --input gs://your-bucket/videos/ \
    --project your-gcp-project \
    --prompt-template grooming \
    --output-dir ./results
```

### Process a single GCS video:
```bash
python label_mouse_behavior_with_gemini.py \
    --input gs://your-bucket/video.mp4 \
    --project your-gcp-project \
    --prompt-template grooming \
    --output-dir ./results
```

## Advanced Options

### Custom timeout and retry settings:
```bash
python label_mouse_behavior_with_gemini.py \
    --input /path/to/videos \
    --project your-gcp-project \
    --prompt-template grooming \
    --timeout 900 \
    --max-retries 5 \
    --output-dir ./results
```

### With example clips and annotations:
```bash
python label_mouse_behavior_with_gemini.py \
    --input /path/to/videos \
    --project your-gcp-project \
    --prompt-template grooming \
    --example-clips /path/to/example/clips \
    --full-example-annotation /path/to/example.json \
    --full-example-video gs://bucket/example.mp4 \
    --output-dir ./results
```

## Dependencies

For GCS support, install the Google Cloud Storage library:
```bash
pip install google-cloud-storage
```

## Output

The script creates JSON files in the output directory with the format:
`{video_name}_{model_id}.json`

Each file contains the behavior segments identified by the Gemini model. 