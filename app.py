import os
import subprocess
import time
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Optional

import spacy
import gradio as gr
from transformers import pipeline
import torch

# ——— spaCy setup for HF Spaces ———
def setup_spacy():
    """Setup spaCy model with proper error handling for HF Spaces"""
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError:
        print("Downloading spaCy model...")
        try:
            from spacy.cli import download as spacy_download
            spacy_download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
            return nlp
        except Exception as e:
            print(f"Failed to download spaCy model: {e}")
            return None

nlp = setup_spacy()


def retry_on_rate_limit(func, max_retries=2, initial_delay=3, backoff=1.5):
    def wrapper(*args, **kwargs):
        delay = initial_delay
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if "rate limit" in str(e).lower() or "429" in str(e):
                    if attempt < max_retries - 1:
                        print(f"Rate limit detected, retrying in {delay}s...")
                        time.sleep(delay)
                        delay *= backoff
                    else:
                        print("Maximum retries reached for rate limit.")
                        raise
                else:
                    # For non-rate-limit errors, raise immediately
                    raise
    return wrapper


def check_ffmpeg():
    """Check if ffmpeg is available in HF Spaces"""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def chunk_video(input_path: str, chunk_length: int = 180, output_dir: str = None) -> List[Path]:
    """Chunk video with temporary directory handling for HF Spaces"""
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="chunks_")
    
    Path(output_dir).mkdir(exist_ok=True)
    output_pattern = os.path.join(output_dir, "chunk_%03d.mp4")
    
    try:
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-f", "segment", "-segment_time", str(chunk_length),
            "-reset_timestamps", "1", "-c", "copy",
            output_pattern
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            return []
            
        return sorted(Path(output_dir).glob("chunk_*.mp4"))
    except subprocess.TimeoutExpired:
        print("Video chunking timed out")
        return []
    except Exception as e:
        print(f"Error chunking video: {str(e)}")
        return []


def extract_audio(video_path: str, audio_path: str) -> bool:
    """Extract audio with better error handling for HF Spaces"""
    try:
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1",
            "-t", "180",  # Limit to 3 minutes per chunk
            audio_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            print(f"Audio extraction error: {result.stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print("Audio extraction timed out")
        return False
    except Exception as e:
        print(f"Error extracting audio: {str(e)}")
        return False


def extract_key_phrases(text: str, top_n: int = 5) -> List[str]:
    """Extract key phrases with fallback if spaCy is not available"""
    if nlp is None:
        # Fallback: simple word extraction
        words = text.split()
        key_words = [w for w in words if len(w) > 4 and w.isalpha()]
        return list(dict.fromkeys(key_words))[:top_n]
    
    try:
        doc = nlp(text)
        phrases = [chunk.text.strip() for chunk in doc.noun_chunks if len(chunk.text.strip()) > 2]
        seen = set()
        unique_phrases = [p for p in phrases if not (p.lower() in seen or seen.add(p.lower()))]
        return unique_phrases[:top_n]
    except Exception as e:
        print(f"Error extracting key phrases: {str(e)}")
        return []


def extract_frame(video_path: str, timestamp: str, output_path: str) -> bool:
    """Extract frame with timeout for HF Spaces"""
    try:
        cmd = ["ffmpeg", "-y", "-i", video_path, "-ss", timestamp, "-frames:v", "1", "-q:v", "2", output_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        
        if result.returncode != 0:
            return False
        return True
    except (subprocess.TimeoutExpired, Exception):
        return False


@retry_on_rate_limit
def transcribe_audio(asr_pipeline, audio_path: str) -> List[Dict]:
    """Transcribe audio with improved error handling"""
    try:
        # Use the pipeline with proper parameters
        result = asr_pipeline(
            audio_path,
            return_timestamps=True,
            chunk_length_s=30,
            stride_length_s=5
        )
        
        if isinstance(result, dict):
            if "chunks" in result:
                return result["chunks"]
            else:
                # Handle single result
                text = result.get("text", "")
                timestamps = result.get("timestamps", [(0.0, 30.0)])
                if isinstance(timestamps, list) and len(timestamps) > 0:
                    return [{"text": text, "timestamp": timestamps[0]}]
                else:
                    return [{"text": text, "timestamp": (0.0, 30.0)}]
        elif isinstance(result, list):
            # Handle list of results
            segments = []
            for i, item in enumerate(result):
                if isinstance(item, dict):
                    segments.append({
                        "text": item.get("text", ""),
                        "timestamp": item.get("timestamp", (i*30, (i+1)*30))
                    })
            return segments
        else:
            return [{"text": str(result), "timestamp": (0.0, 30.0)}]
            
    except Exception as e:
        print(f"Transcription error: {str(e)}")
        return [{"text": "Transcription failed", "timestamp": (0.0, 30.0)}]


@retry_on_rate_limit
def summarize_text(summarizer_pipeline, text: str) -> str:
    """Summarize text with proper length handling"""
    if not text.strip():
        return "No content to summarize."
    
    # Clean and prepare text
    text = text.strip()
    words = text.split()
    
    # Skip very short texts
    if len(words) < 10:
        return text  # Return original if too short
    
    # Truncate if too long
    if len(words) > 500:
        text = " ".join(words[:500])
    
    try:
        # Calculate appropriate lengths
        input_length = len(words)
        max_new_tokens = min(100, max(20, input_length // 3))
        min_length = min(15, max(5, input_length // 8))
        
        result = summarizer_pipeline(
            text,
            max_new_tokens=max_new_tokens,
            min_length=min_length,
            do_sample=False,
            early_stopping=True
        )
        
        if isinstance(result, list) and len(result) > 0:
            summary = result[0]["summary_text"].strip()
            return summary if summary else text
        return text
        
    except Exception as e:
        print(f"Summarization error: {str(e)}")
        return text  # Return original text if summarization fails


def format_timestamp(seconds: float) -> str:
    """Format seconds into MM:SS format"""
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes:02d}:{remaining_seconds:02d}"


def run_pipeline(video_file: str, progress=gr.Progress()) -> List[Dict]:
    """Main pipeline function optimized for HF Spaces"""
    if not video_file:
        return [{"error": "No video file provided"}]
    
    # Check if ffmpeg is available
    if not check_ffmpeg():
        return [{"error": "FFmpeg is not available in this environment"}]
    
    progress(0.1, desc="Initializing models...")
    
    # Initialize models with proper configuration
    try:
        # Configure Whisper with proper settings
        asr = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-tiny",  # Use tiny model for better compatibility
            device=0 if torch.cuda.is_available() else -1,
            model_kwargs={
                "attn_implementation": "eager"  # Fix attention implementation warning
            }
        )
        progress(0.2, desc="ASR model loaded...")
        
        # Configure BART with proper settings
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=0 if torch.cuda.is_available() else -1
        )
        progress(0.3, desc="Summarization model loaded...")
        
    except Exception as e:
        return [{"error": f"Failed to load models: {str(e)}"}]

    # Create temporary directories
    temp_dir = tempfile.mkdtemp(prefix="lecture_capture_")
    chunks_dir = os.path.join(temp_dir, "chunks")
    frames_dir = os.path.join(temp_dir, "frames")
    
    try:
        Path(chunks_dir).mkdir(exist_ok=True)
        Path(frames_dir).mkdir(exist_ok=True)
        
        progress(0.4, desc="Processing video chunks...")
        
        # Process video with shorter chunks
        chunks = chunk_video(video_file, chunk_length=180, output_dir=chunks_dir)
        if not chunks:
            return [{"error": "No video chunks were created. Video may be corrupted or unsupported format."}]
        
        # Limit number of chunks for HF Spaces
        chunks = chunks[:5]  # Process max 5 chunks (15 minutes)
        
        progress(0.5, desc=f"Processing {len(chunks)} chunks...")
        
        # Process each chunk
        all_segments = []
        for i, chunk in enumerate(chunks):
            progress(0.5 + (0.3 * i / len(chunks)), desc=f"Processing chunk {i+1}/{len(chunks)}...")
            
            wav_path = str(chunk).replace(".mp4", ".wav")
            
            # Extract audio
            if not extract_audio(str(chunk), wav_path):
                print(f"Failed to extract audio from chunk {i}")
                continue
            
            # Transcribe with better error handling
            try:
                chunk_segments = transcribe_audio(asr, wav_path)
                
                # Calculate absolute timestamps
                chunk_start_time = i * 180  # 180 seconds per chunk
                
                for seg in chunk_segments:
                    timestamp = seg.get("timestamp", (0.0, 30.0))
                    if isinstance(timestamp, tuple) and len(timestamp) == 2:
                        start_time = chunk_start_time + timestamp[0]
                        end_time = chunk_start_time + timestamp[1]
                    else:
                        start_time = chunk_start_time
                        end_time = chunk_start_time + 30
                    
                    text = seg.get("text", "").strip()
                    if text:  # Only add non-empty segments
                        all_segments.append({
                            "text": text,
                            "start": format_timestamp(start_time),
                            "end": format_timestamp(end_time),
                            "start_seconds": start_time,
                            "end_seconds": end_time
                        })
                        
            except Exception as e:
                print(f"Error processing chunk {i}: {str(e)}")
                continue
            
            # Clean up audio file immediately
            try:
                os.remove(wav_path)
            except:
                pass
        
        if not all_segments:
            return [{"error": "No segments were successfully processed"}]
        
        progress(0.8, desc="Generating summaries and extracting key phrases...")
        
        # Sort segments by start time
        all_segments.sort(key=lambda x: x["start_seconds"])
        
        # Generate timeline (limit to 15 segments for HF Spaces)
        timeline = []
        for i, segment in enumerate(all_segments[:15]):
            segment_text = segment["text"]
            
            # Generate summary
            try:
                summary = summarize_text(summarizer, segment_text) if len(segment_text.split()) > 5 else segment_text
            except Exception as e:
                summary = segment_text
            
            # Extract key phrases
            key_phrases = extract_key_phrases(segment_text) if segment_text else []
            
            timeline.append({
                "segment": i + 1,
                "start_time": segment["start"],
                "end_time": segment["end"],
                "text": segment_text,
                "summary": summary,
                "key_phrases": key_phrases
            })
        
        progress(1.0, desc="Processing complete!")
        return timeline
    
    except Exception as e:
        import traceback
        return [{"error": f"Pipeline failed: {str(e)}", "details": traceback.format_exc()}]
    
    finally:
        # Clean up temporary files
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Failed to clean up temp directory: {str(e)}")


# ——— Gradio UI optimized for HF Spaces ———
def create_interface():
    with gr.Blocks(title="Lecture Capture AI Pipeline", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎓 Lecture Capture AI Pipeline
        
        Upload a lecture video to automatically generate:
        - 📝 Transcription with timestamps
        - 📋 Summaries for each segment  
        - 🔑 Key phrases extraction
        
        **Note**: Optimized for Hugging Face Spaces. Processing limited to 15 minutes of video.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(
                    label="📹 Upload Lecture Video",
                    height=300
                )
                
                process_btn = gr.Button(
                    "🚀 Process Video", 
                    variant="primary",
                    size="lg"
                )
                
                gr.Markdown("""
                ### 💡 Tips:
                - Videos up to 15 minutes work best
                - Clear audio improves transcription quality
                - Processing takes 2-5 minutes
                - Supported formats: MP4, AVI, MOV
                """)
            
            with gr.Column(scale=2):
                output_json = gr.JSON(
                    label="📊 Generated Timeline",
                    height=600
                )
        
        process_btn.click(
            fn=run_pipeline,
            inputs=[video_input],
            outputs=[output_json],
            show_progress=True
        )
        
        gr.Markdown("""
        ### 🔧 Technical Details:
        - Uses Whisper (tiny) for speech recognition
        - BART for text summarization  
        - spaCy for key phrase extraction
        - Optimized for Hugging Face Spaces environment
        """)
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
