"""Demonstration script for audio transcription using faster-whisper.

This version uses the faster-whisper library which offers better performance than whisper.cpp.
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from rich import box
from rich.console import Console
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.rule import Rule
from rich.table import Table

# Add the project root to the Python path
# This allows finding the ultimate package when running the script directly
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

EXAMPLE_DIR = Path(__file__).parent
DATA_DIR = EXAMPLE_DIR / "data"
SAMPLE_AUDIO_PATH = str(DATA_DIR / "Steve_Jobs_Introducing_The_iPhone_compressed.mp3")

from ultimate_mcp_server.utils import get_logger  # noqa: E402

# --- Configuration ---
logger = get_logger("audio_demo")

# Get the directory of the current script
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / "data"

# Define allowed audio extensions
ALLOWED_EXTENSIONS = [".mp3", ".wav", ".flac", ".ogg", ".m4a"]


# --- Helper Functions ---
def find_audio_files(directory: Path) -> List[Path]:
    """Finds audio files with allowed extensions in the given directory."""
    return [
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS
    ]


def format_timestamp(seconds: float) -> str:
    """Format seconds into a timestamp string."""
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"
    else:
        return f"{minutes:02d}:{secs:05.2f}"


def detect_device() -> Tuple[str, str, str]:
    """Detect if CUDA GPU is available and return appropriate device and compute_type."""
    try:
        # Import torch to check if CUDA is available
        import torch

        if torch.cuda.is_available():
            # Get GPU info for display
            gpu_name = torch.cuda.get_device_name(0)
            return "cuda", "float16", gpu_name
        else:
            return "cpu", "int8", None
    except ImportError:
        # If torch is not available, try to directly check for NVIDIA GPUs with ctranslate2
        try:
            import subprocess

            nvidia_smi_output = subprocess.check_output(
                ["nvidia-smi", "-L"], text=True, stderr=subprocess.DEVNULL
            )
            if "GPU" in nvidia_smi_output:
                # Extract GPU name
                gpu_name = nvidia_smi_output.strip().split(":")[1].strip().split("(")[0].strip()
                return "cuda", "float16", gpu_name
            else:
                return "cpu", "int8", None
        except Exception:
            # If all else fails, default to CPU
            return "cpu", "int8", None


def generate_markdown_transcript(transcript: Dict[str, Any], file_path: str) -> str:
    """Generate a markdown version of the transcript with metadata."""
    audio_filename = os.path.basename(file_path)
    metadata = transcript.get("metadata", {})
    segments = transcript.get("segments", [])

    markdown = [
        f"# Transcript: {audio_filename}",
        "",
        "## Metadata",
        f"- **Duration:** {format_timestamp(metadata.get('duration', 0))}",
        f"- **Language:** {metadata.get('language', 'unknown')} (confidence: {metadata.get('language_probability', 0):.2f})",
        f"- **Transcription Model:** {metadata.get('model', 'unknown')}",
        f"- **Device:** {metadata.get('device', 'unknown')}",
        f"- **Processing Time:** {transcript.get('processing_time', {}).get('total', 0):.2f} seconds",
        "",
        "## Full Transcript",
        "",
        transcript.get("enhanced_transcript", transcript.get("raw_transcript", "")),
        "",
        "## Segments",
        "",
    ]

    for segment in segments:
        start_time = format_timestamp(segment["start"])
        end_time = format_timestamp(segment["end"])
        markdown.append(f"**[{start_time} → {end_time}]** {segment['text']}")
        markdown.append("")

    return "\n".join(markdown)


def save_markdown_transcript(transcript: Dict[str, Any], file_path: str) -> Tuple[str, str]:
    """Save the transcript as markdown and text files.

    Returns:
        Tuple containing paths to markdown and text files
    """
    audio_path = Path(file_path)
    markdown_path = audio_path.with_suffix(".md")
    txt_path = audio_path.with_suffix(".txt")

    # Generate and save markdown (enhanced transcript)
    markdown_content = generate_markdown_transcript(transcript, file_path)
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    # Save raw transcript as plain text file
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(transcript.get("raw_transcript", ""))

    return str(markdown_path), str(txt_path)


async def enhance_transcript_with_llm(raw_transcript: str, console: Console) -> str:
    """Enhance the transcript using an LLM to improve readability."""
    try:
        from ultimate_mcp_server.tools.completion import chat_completion
    except ImportError:
        console.print(
            "[yellow]Ultimate MCP Server tools not available for enhancement. Using raw transcript.[/yellow]"
        )
        return raw_transcript

    # Setup progress display
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]Enhancing transcript with LLM[/bold green]"),
        BarColumn(),
        TextColumn("[cyan]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        enhance_task = progress.add_task("Enhancing...", total=100)

        try:
            # Create the prompt for transcript enhancement
            system_prompt = """You are an expert transcription editor. Your task is to enhance the following raw transcript:
1. Fix any spelling or grammar errors
2. Add proper punctuation and capitalization
3. Format the text into logical paragraphs
4. Remove filler words and repeated phrases
5. Preserve the original meaning and all factual content
6. Format numbers, acronyms, and technical terms consistently
7. Keep the text faithful to the original but make it more readable"""

            user_prompt = f"Here is the raw transcript to enhance:\n\n{raw_transcript}\n\nPlease provide only the enhanced transcript without explanations."

            # Split the transcript into chunks if it's very long
            progress.update(enhance_task, completed=20)

            # Call the chat completion function
            result = await chat_completion(
                system_prompt=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                model="gpt-4.1-mini",
                temperature=0.3,
            )

            progress.update(enhance_task, completed=90)

            enhanced_transcript = result.get("content", raw_transcript)

            progress.update(enhance_task, completed=100)

            return enhanced_transcript

        except Exception as e:
            console.print(f"[red]Error enhancing transcript: {e}[/red]")
            progress.update(enhance_task, completed=100)
            return raw_transcript


async def transcribe_with_faster_whisper(file_path: str, console: Console) -> Dict[str, Any]:
    """Transcribe audio using faster-whisper library with real-time progress updates."""
    logger.info(f"Processing file: {file_path}")

    # Check if audio file exists
    if not os.path.exists(file_path):
        logger.error(f"Audio file not found at {file_path}")
        return {"success": False, "error": f"Audio file not found at {file_path}"}

    try:
        # Import faster-whisper - install if not present
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            console.print("[yellow]faster-whisper not installed. Installing now...[/yellow]")
            import subprocess

            subprocess.check_call([sys.executable, "-m", "pip", "install", "faster-whisper"])
            from faster_whisper import WhisperModel

        # Start timing
        start_time = time.time()

        # Get audio duration for progress calculation
        audio_duration = 0
        with Progress(
            SpinnerColumn(), TextColumn("[bold blue]{task.description}"), console=console
        ) as progress:
            analysis_task = progress.add_task("Analyzing audio file...", total=None)
            try:
                import av

                with av.open(file_path) as container:
                    # Get duration in seconds
                    if container.duration is not None:
                        audio_duration = container.duration / 1000000  # microseconds to seconds
                        console.print(
                            f"Audio duration: [cyan]{format_timestamp(audio_duration)}[/cyan] seconds"
                        )
                        progress.update(analysis_task, completed=True)
            except Exception as e:
                console.print(f"[yellow]Could not determine audio duration: {e}[/yellow]")

        # Detect device (CPU or GPU)
        device, compute_type, gpu_name = detect_device()

        # Load the model with progress
        model_size = "large-v3"
        console.print(f"Loading Whisper model: [bold]{model_size}[/bold]")

        if device == "cuda" and gpu_name:
            console.print(
                f"Using device: [bold green]GPU ({gpu_name})[/bold green], compute_type: [bold cyan]{compute_type}[/bold cyan]"
            )
        else:
            console.print(
                f"Using device: [bold yellow]CPU[/bold yellow], compute_type: [bold cyan]{compute_type}[/bold cyan]"
            )

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[bold cyan]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            load_task = progress.add_task("Loading model...", total=100)
            model = WhisperModel(
                model_size, device=device, compute_type=compute_type, download_root="./models"
            )
            progress.update(load_task, completed=100)

        # Setup progress display for transcription
        console.print("\n[bold green]Starting transcription...[/bold green]")

        # Create table for displaying transcribed segments in real time
        table = Table(title="Transcription Progress", expand=True, box=box.ROUNDED)
        table.add_column("Segment")
        table.add_column("Time", style="yellow")
        table.add_column("Text", style="white")

        # Progress bar for overall transcription
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Transcribing..."),
            BarColumn(),
            TextColumn("[cyan]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )

        # Add main progress task
        transcribe_task = progress.add_task("Transcription", total=100)

        # Combine table and progress bar
        transcription_display = Table.grid()
        transcription_display.add_row(table)
        transcription_display.add_row(progress)

        segments_list = []
        segment_idx = 0

        # Run the transcription with live updating display
        with Live(transcription_display, console=console, refresh_per_second=10) as live:
            # Run transcription
            segments, info = model.transcribe(
                file_path,
                beam_size=5,
                vad_filter=True,
                word_timestamps=True,
                language="en",  # Specify language to avoid language detection phase
            )

            # Process segments as they become available
            for segment in segments:
                segments_list.append(segment)

                # Update progress bar based on timestamp
                if audio_duration > 0:
                    current_progress = min(int((segment.end / audio_duration) * 100), 99)
                    progress.update(transcribe_task, completed=current_progress)

                # Add segment to table
                timestamp = f"[{format_timestamp(segment.start)} → {format_timestamp(segment.end)}]"
                table.add_row(f"[cyan]#{segment_idx + 1}[/cyan]", timestamp, segment.text)

                # Update the live display
                live.update(transcription_display)
                segment_idx += 1

            # Finish progress
            progress.update(transcribe_task, completed=100)

        # Build full transcript
        raw_transcript = " ".join([segment.text for segment in segments_list])

        # Convert segments to dictionary format
        segments_dict = []
        for segment in segments_list:
            segments_dict.append(
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "words": [
                        {
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                            "probability": word.probability,
                        }
                        for word in (segment.words or [])
                    ],
                }
            )

        # Enhance the transcript with LLM
        console.print(
            "\n[bold green]Raw transcription complete. Now enhancing the transcript...[/bold green]"
        )
        enhanced_transcript = await enhance_transcript_with_llm(raw_transcript, console)

        # Calculate processing time
        processing_time = time.time() - start_time

        # Create the result dictionary
        result = {
            "success": True,
            "raw_transcript": raw_transcript,
            "enhanced_transcript": enhanced_transcript,
            "segments": segments_dict,
            "metadata": {
                "language": info.language,
                "language_probability": info.language_probability,
                "model": model_size,
                "duration": audio_duration,
                "device": device,
            },
            "processing_time": {"total": processing_time, "transcription": processing_time},
        }

        # Save the transcripts
        markdown_path, txt_path = save_markdown_transcript(result, file_path)
        console.print(
            f"\n[bold green]Saved enhanced transcript to:[/bold green] [cyan]{markdown_path}[/cyan]"
        )
        console.print(f"[bold green]Saved raw transcript to:[/bold green] [cyan]{txt_path}[/cyan]")

        return result

    except Exception as e:
        import traceback

        logger.error(f"Transcription error: {e}")
        logger.error(traceback.format_exc())
        return {"success": False, "error": f"Transcription error: {e}"}


async def main():
    """Runs the audio transcription demonstrations."""

    logger.info("Starting Audio Transcription Demo", emoji_key="audio")

    console = Console()
    console.print(Rule("[bold green]Audio Transcription Demo (faster-whisper)[/bold green]"))

    # --- Find Audio Files ---
    audio_files = find_audio_files(DATA_DIR)
    if not audio_files:
        console.print(
            f"[bold red]Error:[/bold red] No audio files found in {DATA_DIR}. Please place audio files (e.g., .mp3, .wav) there."
        )
        return

    console.print(f"Found {len(audio_files)} audio file(s) in {DATA_DIR}:")
    for f in audio_files:
        console.print(f"- [cyan]{f.name}[/cyan]")
    console.print()

    # --- Process Each File ---
    for file_path in audio_files:
        try:
            console.print(
                Panel(
                    f"Processing file: [cyan]{escape(str(file_path))}[/cyan]",
                    title="Audio Transcription",
                    border_style="blue",
                )
            )

            # Call our faster-whisper transcription function
            result = await transcribe_with_faster_whisper(str(file_path), console)

            if result.get("success", False):
                console.print(
                    f"[green]Transcription successful for {escape(str(file_path))}.[/green]"
                )

                # Show comparison of raw vs enhanced transcript
                if "raw_transcript" in result and "enhanced_transcript" in result:
                    comparison = Table(title="Transcript Comparison", expand=True, box=box.ROUNDED)
                    comparison.add_column("Raw Transcript", style="yellow")
                    comparison.add_column("Enhanced Transcript", style="green")

                    # Limit to a preview of the first part
                    raw_preview = result["raw_transcript"][:500] + (
                        "..." if len(result["raw_transcript"]) > 500 else ""
                    )
                    enhanced_preview = result["enhanced_transcript"][:500] + (
                        "..." if len(result["enhanced_transcript"]) > 500 else ""
                    )

                    comparison.add_row(raw_preview, enhanced_preview)
                    console.print(comparison)

                    # Display metadata if available
                    if "metadata" in result and result["metadata"]:
                        console.print("[bold]Metadata:[/bold]")
                        for key, value in result["metadata"].items():
                            console.print(f"  - [cyan]{key}[/cyan]: {value}")

                    # Display processing time
                    if "processing_time" in result:
                        console.print("[bold]Processing Times:[/bold]")
                        for key, value in result["processing_time"].items():
                            if isinstance(value, (int, float)):
                                console.print(f"  - [cyan]{key}[/cyan]: {value:.2f}s")
                            else:
                                console.print(f"  - [cyan]{key}[/cyan]: {value}")
                else:
                    console.print("[yellow]Warning:[/yellow] No transcript was returned.")
            else:
                console.print(
                    f"[bold red]Transcription failed:[/bold red] {escape(result.get('error', 'Unknown error'))}"
                )

            console.print()  # Add a blank line between files

        except Exception as outer_e:
            import traceback

            console.print(
                f"[bold red]Unexpected error processing file {escape(str(file_path))}:[/bold red] {escape(str(outer_e))}"
            )
            console.print("[bold red]Traceback:[/bold red]")
            console.print(escape(traceback.format_exc()))
            continue  # Move to the next file

    logger.info("Audio Transcription Demo Finished", emoji_key="audio")


if __name__ == "__main__":
    # Basic error handling for the async execution itself
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"An error occurred running the demo: {e}")
        import traceback

        traceback.print_exc()
