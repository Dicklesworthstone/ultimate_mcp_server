"""
Tournament task implementations for asynchronous tournament execution.
"""

# Standard Library Imports
import asyncio
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from ultimate_mcp_server.core.evaluation.base import EvaluationScore
from ultimate_mcp_server.core.models.tournament import (
    ModelConfig,
    ModelResponseData,
    TournamentData,
    TournamentRoundResult,
    TournamentStatus,
)
from ultimate_mcp_server.core.tournaments.manager import tournament_manager
from ultimate_mcp_server.core.tournaments.utils import (
    calculate_weighted_score,
    create_round_prompt,
    extract_thinking,
    generate_comparison_file_content,
    generate_leaderboard_file_content,
    save_model_response_content,
    update_overall_best_response,
)
from ultimate_mcp_server.tools.completion import generate_completion
from ultimate_mcp_server.tools.extraction import extract_code_from_response
from ultimate_mcp_server.utils.logging import get_logger

logger = get_logger("ultimate_mcp_server.tournaments.tasks")

# --- Global semaphore for concurrent model calls ---
MODEL_CALL_SEMAPHORE: Optional[asyncio.Semaphore] = None


def initialize_semaphore(max_concurrent_calls: int):
    global MODEL_CALL_SEMAPHORE
    MODEL_CALL_SEMAPHORE = asyncio.Semaphore(max_concurrent_calls)
    logger.info(f"Tournament task semaphore initialized with concurrency: {max_concurrent_calls}")


async def run_tournament_async(tournament_id: str):
    """Main async task to orchestrate the entire tournament."""
    await asyncio.sleep(0.1)  # Small delay for state propagation

    tournament = tournament_manager.get_tournament(tournament_id, force_reload=True)
    if not tournament:
        logger.error(f"[TASK_ERROR] Tournament {tournament_id} not found for execution.")
        return

    if tournament.status != TournamentStatus.RUNNING:  # Check if it was set to RUNNING
        logger.warning(
            f"[TASK_WARN] Tournament {tournament_id} not in RUNNING state ({tournament.status}). Aborting task."
        )
        return

    # --- Initialize semaphore based on tournament config ---
    if (
        MODEL_CALL_SEMAPHORE is None
        or MODEL_CALL_SEMAPHORE._value != tournament.config.max_concurrent_model_calls
    ):
        initialize_semaphore(tournament.config.max_concurrent_model_calls)

    logger.info(
        f"[TASK_START] Starting execution for tournament '{tournament.name}' (ID: {tournament_id})"
    )

    try:
        if tournament.current_round < 0:  # If just started
            tournament.current_round = 0

        while tournament.current_round < tournament.config.rounds:
            # --- Check for cancellation before starting a round ---
            current_tournament_state = tournament_manager.get_tournament(
                tournament_id, force_reload=True
            )
            if (
                not current_tournament_state
                or current_tournament_state.status == TournamentStatus.CANCELLED
            ):
                logger.info(
                    f"[TASK_CANCEL] Tournament {tournament_id} cancelled. Halting execution."
                )
                if (
                    current_tournament_state
                    and current_tournament_state.status != TournamentStatus.CANCELLED
                ):  # Ensure it's marked
                    tournament_manager.update_tournament_status(
                        tournament_id, TournamentStatus.CANCELLED, "Cancelled during execution."
                    )
                return

            round_num = tournament.current_round
            logger.info(
                f"[ROUND_START] Processing Round {round_num}/{tournament.config.rounds - 1} for '{tournament.name}'"
            )

            round_result_obj = tournament.rounds_results[
                round_num
            ]  # Assumes initialized by manager
            round_result_obj.status = TournamentStatus.RUNNING
            round_result_obj.start_time = datetime.now(timezone.utc)
            tournament_manager._save_tournament_state(tournament)

            await process_single_round(tournament, round_num, round_result_obj)

            round_result_obj.status = TournamentStatus.COMPLETED  # Mark round as completed
            round_result_obj.end_time = datetime.now(timezone.utc)
            tournament_manager._save_tournament_state(tournament)
            logger.info(f"[ROUND_END] Round {round_num} for '{tournament.name}' completed.")

            # --- Update overall best response after each round ---
            update_overall_best_response(tournament)  # Utility function to find and set best
            tournament_manager._save_tournament_state(tournament)

            tournament.current_round += 1
            tournament_manager._save_tournament_state(tournament)  # Save progress

        tournament.status = TournamentStatus.COMPLETED
        logger.info(
            f"[TASK_COMPLETE] Tournament '{tournament.name}' (ID: {tournament_id}) completed successfully."
        )

    except Exception as e:
        logger.error(f"[TASK_FAILURE] Tournament '{tournament.name}' failed: {e}", exc_info=True)
        tournament.status = TournamentStatus.FAILED
        tournament.error_message = str(e)
    finally:
        tournament.end_time = datetime.now(timezone.utc)
        tournament_manager._save_tournament_state(tournament)
        logger.info(
            f"Final state saved for tournament {tournament_id}. Status: {tournament.status}"
        )


async def process_single_round(
    tournament: TournamentData, round_num: int, round_result_obj: TournamentRoundResult
):
    """Processes all model variants for a single round."""

    # Determine previous responses for synthesis rounds > 0
    previous_round_variant_responses: Dict[str, ModelResponseData] = {}
    if round_num > 0:
        prev_round_idx = round_num - 1
        if prev_round_idx < len(tournament.rounds_results):
            previous_round_result = tournament.rounds_results[prev_round_idx]
            previous_round_variant_responses = (
                previous_round_result.responses
            )  # These are already ModelResponseData objects
        else:
            logger.warning(
                f"Could not find previous round {prev_round_idx} data for round {round_num}. Proceeding without it."
            )

    tasks = []
    for model_cfg in tournament.config.models:
        for i in range(model_cfg.diversity_count):
            variant_id = f"{model_cfg.model_id}/v{i}"

            # --- Check for cancellation before each model task ---
            current_tournament_state = tournament_manager.get_tournament(
                tournament.tournament_id, force_reload=True
            )
            if (
                not current_tournament_state
                or current_tournament_state.status == TournamentStatus.CANCELLED
            ):
                logger.info(
                    f"[MODEL_TASK_CANCEL] Cancellation detected for tournament {tournament.tournament_id}. Skipping variant {variant_id}."
                )
                continue  # Skip remaining tasks in this round

            # Skip if already processed (e.g., resuming a failed round)
            if (
                variant_id in round_result_obj.responses
                and round_result_obj.responses[variant_id].response_text
            ):
                logger.info(
                    f"Variant {variant_id} for round {round_num} already processed. Skipping."
                )
                continue

            tasks.append(
                process_single_model_variant(
                    tournament,
                    model_cfg,
                    variant_id,  # Pass the unique variant ID
                    round_num,
                    round_result_obj,
                    previous_round_variant_responses,  # Pass full ModelResponseData dict
                )
            )

    if not tasks:
        logger.info(f"No new model variants to process for round {round_num}.")
        round_result_obj.status = TournamentStatus.COMPLETED
        return

    logger.info(f"Gathering {len(tasks)} model variant tasks for round {round_num}.")
    # Await all tasks and catch any unhandled exceptions
    try:
        await asyncio.gather(*tasks)
    except Exception as e:
        logger.error(
            f"An error occurred during asyncio.gather in round {round_num}: {e}", exc_info=True
        )
        round_result_obj.error_message = (
            getattr(round_result_obj, "error_message", "") or ""
        ) + f"; Error during task gathering: {str(e)}"
        round_result_obj.status = TournamentStatus.FAILED  # Mark round as failed if gather fails
        # Individual task errors are handled within process_single_model_variant
        return

    # --- Generate comparison and leaderboard files ---
    # (Ensure these utils exist and are updated)
    comparison_md = generate_comparison_file_content(tournament, round_num)
    leaderboard_md = generate_leaderboard_file_content(tournament, round_num)  # New utility

    round_storage_path = Path(tournament.storage_path) / f"round_{round_num}"
    round_storage_path.mkdir(parents=True, exist_ok=True)

    if comparison_md:
        comp_path = round_storage_path / "round_comparison_report.md"
        comp_path.write_text(comparison_md, encoding="utf-8")
        round_result_obj.comparison_file_path = str(comp_path)
    if leaderboard_md:
        lead_path = round_storage_path / "round_leaderboard.md"
        lead_path.write_text(leaderboard_md, encoding="utf-8")
        round_result_obj.leaderboard_file_path = str(lead_path)

    # Save state after generating reports
    tournament_manager._save_tournament_state(tournament)


async def process_single_model_variant(
    tournament: TournamentData,
    model_config: "ModelConfig",  # Forward ref as string, ModelConfig is imported
    variant_id: str,  # e.g., "openai/gpt-4o/v0"
    round_num: int,
    round_result_obj: TournamentRoundResult,
    previous_round_variant_responses: Dict[str, ModelResponseData],
):
    """Processes a single model variant (handles diversity), including retries and evaluation."""

    # --- Acquire semaphore ---
    if MODEL_CALL_SEMAPHORE:  # Should always be initialized
        await MODEL_CALL_SEMAPHORE.acquire()

    response_data = ModelResponseData(
        model_id_original=model_config.model_id, model_id_variant=variant_id, round_num=round_num
    )
    task_start_time = time.monotonic()

    # --- Prepare storage paths if needed (handled in save_model_response_content) ---

    try:
        # --- Check for cancellation ---
        current_tournament_state = tournament_manager.get_tournament(
            tournament.tournament_id, force_reload=True
        )
        if (
            not current_tournament_state
            or current_tournament_state.status == TournamentStatus.CANCELLED
        ):
            response_data.error = "Tournament cancelled before model execution."
            logger.info(f"Model task {variant_id} skipped due to tournament cancellation.")
            raise asyncio.CancelledError("Tournament cancelled")

        prompt = create_round_prompt(
            tournament,
            round_num,
            previous_round_variant_responses,
            target_model_variant_id=variant_id,  # For personalized prompts if needed
        )

        # --- LLM Call with Retries ---
        current_attempt = 0
        llm_response_dict = None
        while current_attempt <= tournament.config.max_retries_per_model_call:
            try:
                logger.info(
                    f"[MODEL_CALL_START] Attempt {current_attempt + 1}/{tournament.config.max_retries_per_model_call + 1} for {variant_id}, Round {round_num}"
                )

                provider_id = (
                    model_config.model_id.split("/")[0] if "/" in model_config.model_id else None
                )

                # Parameters that are direct arguments to the generate_completion tool
                tool_direct_params = {
                    "prompt": prompt,
                    "model": model_config.model_id,  # Use original model_id for API call
                    "provider": provider_id,
                    "temperature": model_config.temperature,
                    # max_tokens is added conditionally below
                }
                if model_config.max_tokens is not None:
                    tool_direct_params["max_tokens"] = model_config.max_tokens

                # Parameters that should be passed via the 'additional_params' argument of the tool
                tool_additional_params = {}
                if model_config.system_prompt is not None:
                    tool_additional_params["system_prompt"] = model_config.system_prompt
                if model_config.seed is not None:
                    tool_additional_params["seed"] = model_config.seed
                # Example: if model_config had top_p, it would be added here too:
                # if hasattr(model_config, 'top_p') and model_config.top_p is not None:
                #    tool_additional_params["top_p"] = model_config.top_p

                llm_response_dict = await generate_completion(
                    **tool_direct_params, additional_params=tool_additional_params
                )

                if llm_response_dict.get("success"):
                    logger.info(
                        f"[MODEL_CALL_SUCCESS] {variant_id} successful on attempt {current_attempt + 1}"
                    )
                    break  # Success, exit retry loop
                else:
                    error_msg = llm_response_dict.get("error", "Unknown LLM error")
                    logger.warning(
                        f"Attempt {current_attempt + 1} for {variant_id} failed: {error_msg}"
                    )
                    if current_attempt == tournament.config.max_retries_per_model_call:
                        raise RuntimeError(f"LLM call failed after max retries: {error_msg}")

            except Exception as e:  # Catch exceptions from generate_completion itself
                logger.warning(f"Exception on attempt {current_attempt + 1} for {variant_id}: {e}")
                if current_attempt == tournament.config.max_retries_per_model_call:
                    raise RuntimeError(f"LLM call failed after max retries (exception): {e}") from e

            current_attempt += 1
            # Decorrelated jitter backoff
            sleep_time = random.uniform(
                tournament.config.retry_backoff_base_seconds,
                tournament.config.retry_backoff_base_seconds * 1.5 * (2 ** (current_attempt - 1)),
            )
            # Max sleep to prevent overly long waits
            sleep_time = min(sleep_time, 30.0)  # e.g., max 30s backoff
            logger.info(f"Retrying {variant_id} in {sleep_time:.2f} seconds...")
            await asyncio.sleep(sleep_time)

        # --- Process Successful LLM Response ---
        response_data.response_text = llm_response_dict.get("text", "")
        response_data.metrics.update(
            {
                "input_tokens": llm_response_dict.get("tokens", {}).get("input"),
                "output_tokens": llm_response_dict.get("tokens", {}).get("output"),
                "cost": llm_response_dict.get("cost", 0.0),
                "latency_ms": int(llm_response_dict.get("processing_time", 0) * 1000),
                "api_model_id_used": llm_response_dict.get("model", model_config.model_id),
            }
        )

        response_data.thinking_process = await extract_thinking(response_data.response_text)

        if tournament.config.tournament_type == "code":
            # Use the tool function for extraction
            extracted_code_string = await extract_code_from_response(
                response_text=response_data.response_text,
                model=tournament.config.extraction_model_id,  # Pass extraction_model_id as the model for extraction
                # timeout parameter uses its default from extract_code_from_response
            )
            if extracted_code_string:  # Check if a non-empty string was returned
                response_data.extracted_code = extracted_code_string.strip()
            else:
                logger.warning(
                    f"Code extraction returned empty or failed for {variant_id}. Original response length: {len(response_data.response_text or '')}"
                )
                response_data.extracted_code = (
                    None  # Explicitly set to None on failure or empty string
                )

        # --- Save response content ---
        # (This util saves the main readable MD and potentially the raw code file)
        saved_paths = await save_model_response_content(
            tournament_storage_path=Path(tournament.storage_path),
            round_num=round_num,
            variant_id=variant_id,  # Use variant_id for unique filenames
            response_text=response_data.response_text,
            extracted_code=response_data.extracted_code,
            thinking_process=response_data.thinking_process,
            metrics=response_data.metrics,
            tournament_type=tournament.config.tournament_type,
        )
        response_data.response_file_path = saved_paths.get("markdown_file")
        response_data.extracted_code_file_path = saved_paths.get("code_file")

        # --- Run Evaluations ---
        evaluators = tournament_manager.get_evaluators_for_tournament(tournament.tournament_id)
        if evaluators:
            logger.info(f"Running {len(evaluators)} evaluators for {variant_id}...")
            for evaluator_instance in evaluators:
                eval_config = next(
                    (
                        e
                        for e in tournament.config.evaluators
                        if e.evaluator_id
                        == evaluator_instance.config.get(
                            "evaluator_id_ref", evaluator_instance.evaluator_type
                        )
                    ),
                    None,
                )  # Find original config for ID

                eval_id_for_scores = (
                    eval_config.evaluator_id if eval_config else evaluator_instance.evaluator_type
                )

                try:
                    eval_score_obj = await evaluator_instance.score(
                        response_data,  # Pass the full ModelResponseData
                        tournament.config.prompt,
                        tournament.config.tournament_type,
                    )
                    response_data.scores[eval_id_for_scores] = (
                        eval_score_obj.model_dump()
                    )  # Store full score object
                    logger.debug(
                        f"Evaluator '{eval_id_for_scores}' score for {variant_id}: {eval_score_obj.score}"
                    )
                except Exception as eval_e:
                    logger.error(
                        f"Evaluator '{eval_id_for_scores}' failed for {variant_id}: {eval_e}",
                        exc_info=True,
                    )
                    response_data.scores[eval_id_for_scores] = EvaluationScore(
                        score=0.0, details=f"Evaluation error: {str(eval_e)}"
                    ).model_dump()

            # Calculate overall weighted score
            response_data.overall_score = calculate_weighted_score(
                response_data.scores, tournament.config.evaluators
            )

    except asyncio.CancelledError:  # Handle task cancellation gracefully
        logger.info(f"Task for {variant_id} in round {round_num} was cancelled.")
        response_data.error = "Task cancelled."
        response_data.metrics["final_status"] = "cancelled"
    except Exception as e:
        logger.error(f"[MODEL_TASK_FAILURE] Error processing {variant_id}: {e}", exc_info=True)
        response_data.error = str(e)
        response_data.metrics["final_status"] = "failed"
    finally:
        response_data.metrics["total_task_time_ms"] = int(
            (time.monotonic() - task_start_time) * 1000
        )
        # --- Add response to the round_result_obj (which is part of tournament state) ---
        # This needs to be thread-safe if multiple tasks could update this concurrently,
        # but asyncio tasks run on a single thread, so direct assignment is fine here.
        # The `tournament` object itself is shared, so saving it needs care.
        round_result_obj.responses[variant_id] = response_data

        # Defer saving the full tournament state to the calling round processor
        # to batch saves, but log that this variant is done.
        logger.info(
            f"Finished processing variant {variant_id}. Error: {response_data.error is not None}"
        )

        # --- Release semaphore ---
        if MODEL_CALL_SEMAPHORE:
            MODEL_CALL_SEMAPHORE.release()


async def process_single_model(
    model_id: str,
    prompt: str,
    tournament_id: str,
    round_num: int,
    is_code_tournament: bool,
    extraction_model_id: Optional[str] = None,
) -> ModelResponseData:
    """
    Handles the logic for calling a single model provider using the generate_completion tool.
    """
    start_time = time.monotonic()
    logger.info(f"[MODEL TASK] Processing model {model_id} for round {round_num}")

    # Get tournament to access storage path
    tournament = tournament_manager.get_tournament(tournament_id)
    if not tournament:
        raise ValueError(f"Tournament {tournament_id} not found")

    # Setup storage paths
    round_storage_path = Path(tournament.storage_path) / f"round_{round_num}"
    round_storage_path.mkdir(exist_ok=True, parents=True)

    response_data = ModelResponseData(
        model_id_original=model_id,
        model_id_variant=model_id,  # In this context, variant is the same as original
        round_num=round_num,
    )
    extracted_code: Optional[str] = None  # noqa: F841
    file_extension = ".py" if is_code_tournament else ".md"

    provider_name = (
        model_id.split("/")[0] if "/" in model_id else None
    )  # Infer provider from model_id if possible
    if not provider_name:
        logger.warning(
            f"[MODEL TASK] Could not infer provider from model_id: {model_id}. Attempting call without explicit provider."
        )
        # Note: generate_completion might fail if provider isn't specified and cannot be inferred

    try:
        # Use generate_completion tool
        logger.info(
            f"[MODEL TASK] Calling generate_completion for model {model_id} with prompt length {len(prompt)}"
        )
        # Log prompt preview
        preview_length = 100
        prompt_preview = prompt[:preview_length] + "..." if len(prompt) > preview_length else prompt
        logger.info(f"[MODEL TASK] Prompt preview: {prompt_preview}")

        # Call the tool function directly
        completion_result_dict = await generate_completion(
            prompt=prompt,
            model=model_id,  # Pass the full model ID
            provider=provider_name,  # Pass inferred provider
            # Add other params like max_tokens, temperature if needed/available in TournamentConfig
        )

        # Check for success
        if not completion_result_dict.get("success"):
            error_msg = completion_result_dict.get(
                "error", "generate_completion tool indicated failure"
            )
            raise RuntimeError(f"Completion failed for {model_id}: {error_msg}")

        # Extract data from the dictionary returned by the tool
        response_text = completion_result_dict.get("text", "")
        actual_model_used = completion_result_dict.get(
            "model", model_id
        )  # Use actual model if returned
        token_info = completion_result_dict.get("tokens", {})
        cost = completion_result_dict.get("cost", 0.0)
        processing_time_sec = completion_result_dict.get("processing_time", 0.0)
        latency_ms = int(processing_time_sec * 1000)

        # Log response preview
        response_preview = (
            response_text[:preview_length] + "..."
            if len(response_text) > preview_length
            else response_text
        )
        logger.info(f"[MODEL TASK] Response preview for {actual_model_used}: {response_preview}")

        # Extract metrics from the tool result
        completion_metrics = {
            "input_tokens": token_info.get("input"),
            "output_tokens": token_info.get("output"),
            "cost": cost,
            "latency_ms": latency_ms,  # Use processing_time from tool
            "api_model_id_used": actual_model_used,  # Store the actual model ID used by the API
        }

        # Process response - use async extract_thinking
        thinking = await extract_thinking(response_text)
        code_metrics = {}  # Placeholder for potential future code analysis metrics

        # Save response to file with better naming pattern
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe_model_id = re.sub(r"[^a-zA-Z0-9_\-.]", "_", actual_model_used)  # Use actual model name
        safe_tournament_id = re.sub(r"[^a-zA-Z0-9_\-.]", "_", tournament_id)

        filename_base = (
            f"tournament_{safe_tournament_id}_round-{round_num}_model-{safe_model_id}_{timestamp}"
        )
        raw_response_path = round_storage_path / f"{filename_base}{file_extension}"

        raw_response_path.write_text(response_text or "", encoding="utf-8")

        # Create a more user-friendly version with added context
        readable_content = f"""# Tournament Response
**Tournament ID:** {tournament_id}
**Round:** {round_num}
**Model (Configured):** {model_id}
**Model (Actual API):** {actual_model_used}
**Timestamp:** {datetime.now().isoformat()}
**Tokens:** {completion_metrics.get("input_tokens", "N/A")} in, {completion_metrics.get("output_tokens", "N/A")} out
**Cost:** ${completion_metrics.get("cost", 0.0):.6f}
**Latency:** {completion_metrics.get("latency_ms", "N/A")}ms

## Prompt
```
{prompt}
```

## Response
```
{response_text}
```
"""
        readable_path = round_storage_path / f"{filename_base}_readable{file_extension}"
        readable_path.write_text(readable_content, encoding="utf-8")

        logger.info(f"[MODEL TASK] Saved response to: {readable_path}")

        # Populate response data
        # model_id_original and model_id_variant are already set
        response_data.response_text = response_text
        response_data.thinking_process = thinking
        response_data.metrics = {**completion_metrics, **code_metrics}
        response_data.timestamp = datetime.now(timezone.utc)
        response_data.response_file_path = str(raw_response_path)  # Store path to raw response
        response_data.metrics["total_processing_time_ms"] = int(
            (time.monotonic() - start_time) * 1000
        )  # Keep overall task time

        logger.info(
            f"[MODEL TASK] Finished processing model {actual_model_used} for round {round_num} in {response_data.metrics['total_processing_time_ms']}ms"
        )

    except Exception as e:
        logger.error(f"[MODEL TASK] Error processing model {model_id}: {e}", exc_info=True)
        response_data.error = str(e)

    return response_data


async def run_single_round_task(tournament_id: str, round_num: int):
    """
    Task that runs a single round of the tournament, including LLM calls.
    """
    logger.info(f"[ROUND TASK START] Running round {round_num} for tournament {tournament_id}")
    tournament = tournament_manager.get_tournament(tournament_id, force_reload=True)

    # --- Check if tournament exists or was cancelled before proceeding ---
    if not tournament:
        logger.error(
            f"[ROUND TASK FAIL] Tournament {tournament_id} not found at start of round {round_num}."
        )
        return
    if tournament.status == TournamentStatus.CANCELLED:
        logger.info(
            f"[ROUND TASK ABORT] Tournament {tournament_id} was cancelled. Stopping round {round_num}."
        )
        # Ensure round status reflects cancellation if it was running
        if round_num < len(tournament.rounds_results):
            round_result = tournament.rounds_results[round_num]
            if round_result.status == TournamentStatus.RUNNING:
                round_result.status = TournamentStatus.CANCELLED
                round_result.error = "Cancelled by user request during execution."
                round_result.end_time = datetime.now(timezone.utc)
                tournament_manager._save_tournament_state(tournament)
        return
    # -------------------------------------------------------------------

    if round_num >= len(tournament.rounds_results):
        logger.error(
            f"[ROUND TASK FAIL] Invalid round number {round_num} for tournament {tournament_id} state."
        )
        return

    round_result = tournament.rounds_results[round_num]

    try:
        # Mark round as running
        round_result.status = TournamentStatus.RUNNING
        round_result.start_time = datetime.now(timezone.utc)
        tournament_manager._save_tournament_state(tournament)
        logger.info(f"[ROUND TASK] Round {round_num} marked as running")

        # Get tournament config
        is_code_tournament = tournament.config.tournament_type == "code"
        extraction_model_id = tournament.config.extraction_model_id

        # Create prompt for this round
        prompt = create_round_prompt(tournament, round_num)

        # Create tasks for all configured models
        model_tasks = []
        for model_config in tournament.config.models:
            model_id = model_config.model_id

            # Skip if already processed
            if model_id in round_result.responses:
                logger.info(f"[ROUND TASK] Skipping already processed model {model_id}")
                continue

            # Add task for this model
            task = process_single_model(
                model_id=model_id,
                prompt=prompt,
                tournament_id=tournament_id,
                round_num=round_num,
                is_code_tournament=is_code_tournament,
                extraction_model_id=extraction_model_id,
            )
            model_tasks.append(task)
            logger.info(f"[ROUND TASK] Added task for model {model_id}")

        # Exit if no tasks to run
        if not model_tasks:
            logger.info(f"[ROUND TASK] No models to process for round {round_num}")
            round_result.status = TournamentStatus.COMPLETED
            round_result.end_time = datetime.now(timezone.utc)
            tournament_manager._save_tournament_state(tournament)
            return

        # Run all model tasks in parallel
        logger.info(f"[ROUND TASK] Running {len(model_tasks)} model tasks in parallel")
        results = await asyncio.gather(*model_tasks, return_exceptions=True)

        # Process results
        for i, result in enumerate(results):
            model_id = tournament.config.models[i].model_id

            # Handle exceptions
            if isinstance(result, Exception):
                logger.error(
                    f"[ROUND TASK] Error processing model {model_id}: {result}", exc_info=True
                )
                continue

            # Store result
            round_result.responses[model_id] = result
            tournament_manager._save_tournament_state(tournament)

        # Create comparison file
        comparison_content = generate_comparison_file_content(tournament, round_num)
        if comparison_content:
            round_dir = Path(tournament.storage_path) / f"round_{round_num}"
            round_dir.mkdir(exist_ok=True)
            comparison_file = round_dir / "model_comparison.md"

            with open(comparison_file, "w", encoding="utf-8") as f:
                f.write(comparison_content)

            # Store the path in round results
            round_result.comparison_file_path = str(comparison_file)
            tournament_manager._save_tournament_state(tournament)

        # Mark round as completed
        round_result.status = TournamentStatus.COMPLETED
        round_result.end_time = datetime.now(timezone.utc)
        tournament_manager._save_tournament_state(tournament)
        logger.info(
            f"[ROUND TASK COMPLETE] Round {round_num} for tournament {tournament_id} completed successfully"
        )

        # If this was the last round, mark the tournament as completed
        if round_num == tournament.config.rounds - 1:
            tournament.status = TournamentStatus.COMPLETED
            tournament.end_time = datetime.now(timezone.utc)
            tournament_manager._save_tournament_state(tournament)
            logger.info(
                f"[ROUND TASK] Tournament {tournament_id} marked as completed after final round"
            )

    except Exception as e:
        logger.error(f"[ROUND TASK ERROR] Error processing round {round_num}: {e}", exc_info=True)
        round_result.status = TournamentStatus.FAILED
        round_result.error = str(e)
        round_result.end_time = datetime.now(timezone.utc)
        tournament_manager._save_tournament_state(tournament)

        # Mark tournament as failed
        tournament.status = TournamentStatus.FAILED
        tournament.error_message = f"Failed during round {round_num}: {str(e)}"
        tournament.end_time = datetime.now(timezone.utc)
        tournament_manager._save_tournament_state(tournament)


async def process_model_task(
    tournament: TournamentData,
    model_id: str,
    round_num: int,
    previous_round_responses: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Process a single model task for the tournament using generate_completion tool.

    Args:
        tournament: Tournament data
        model_id: Model to use (e.g., 'openai/gpt-4o')
        round_num: Current round number
        previous_round_responses: Previous round responses (for rounds > 0)

    Returns:
        Model task result with response text and metrics
    """
    start_task_time = time.monotonic()
    # Infer provider from model_id format 'provider:model_name' or 'provider/model_name'
    provider_id = None
    if ":" in model_id:
        provider_id = model_id.split(":")[0]
    elif "/" in model_id:  # Keep backward compatibility if '/' is used
        provider_id = model_id.split("/")[0]

    if not provider_id:
        logger.warning(
            f"[MODEL TASK] Could not infer provider from model_id: {model_id}. Attempting call without explicit provider."
        )

    try:
        logger.info(
            f"[MODEL TASK] Processing model {model_id} for round {round_num} (Provider: {provider_id})"
        )

        # Generate prompt based on tournament type and round
        if round_num == 0:
            prompt = tournament.config.prompt
        else:
            prompt = create_round_prompt(tournament, round_num, previous_round_responses)

        # Generate completion using the tool
        logger.info(
            f"[MODEL TASK] Calling generate_completion for model {model_id} with prompt length {len(prompt)}"
        )
        preview_length = 100
        prompt_preview = prompt[:preview_length] + "..." if len(prompt) > preview_length else prompt
        logger.info(f"[MODEL TASK] Prompt preview: {prompt_preview}")

        completion_result_dict = await generate_completion(
            prompt=prompt,
            model=model_id,
            provider=provider_id,  # Pass the inferred provider
            # Add other params like max_tokens, temperature if needed/available
        )

        # Check for success
        if not completion_result_dict.get("success"):
            error_msg = completion_result_dict.get(
                "error", "generate_completion tool indicated failure"
            )
            raise RuntimeError(f"Completion failed for {model_id}: {error_msg}")

        # Extract data from the result dictionary
        response_text = completion_result_dict.get("text", "")
        actual_model_used = completion_result_dict.get("model", model_id)
        token_info = completion_result_dict.get("tokens", {})
        cost = completion_result_dict.get("cost", 0.0)
        processing_time_sec = completion_result_dict.get("processing_time", 0.0)

        # Log response preview
        response_preview = (
            response_text[:preview_length] + "..."
            if len(response_text) > preview_length
            else response_text
        )
        logger.info(f"[MODEL TASK] Response preview for {actual_model_used}: {response_preview}")

        # Extract metrics from the tool result
        completion_metrics = {
            "input_tokens": token_info.get("input"),
            "output_tokens": token_info.get("output"),
            "cost": cost,
            "processing_time_ms": int(processing_time_sec * 1000),  # Use tool's processing time
        }

        # Extract thinking/reasoning if present - use async extract_thinking
        thinking = await extract_thinking(response_text)

        # Save response to a file with timestamp - use async save_model_response
        response_file = await save_model_response_content(
            tournament_storage_path=Path(tournament.storage_path),
            round_num=round_num,
            variant_id=model_id,  # Use model_id for unique filenames
            response_text=response_text,
            extracted_code=None,  # No extracted code for this task
            thinking_process=thinking,
            metrics=completion_metrics,
            tournament_type=tournament.config.tournament_type,
        )

        total_task_time_ms = int((time.monotonic() - start_task_time) * 1000)
        completion_metrics["total_task_time_ms"] = total_task_time_ms  # Add overall task time

        logger.info(
            f"[MODEL TASK] Finished processing model {actual_model_used} for round {round_num} in {total_task_time_ms}ms (LLM time: {completion_metrics['processing_time_ms']}ms)"
        )

        return {
            "model_id": actual_model_used,  # Return actual model used
            "response_text": response_text,
            "thinking": thinking,
            "metrics": completion_metrics,
            "response_file": str(response_file.get("markdown_file"))
            if isinstance(response_file, dict)
            else str(response_file),  # Ensure path is string
        }
    except Exception as e:
        logger.error(f"[MODEL TASK] Error processing model {model_id}: {str(e)}", exc_info=True)
        total_task_time_ms = int((time.monotonic() - start_task_time) * 1000)
        return {
            "model_id": model_id,
            "error": str(e),
            "response_text": f"Error generating response: {str(e)}",
            "thinking": None,
            "metrics": {
                "error": str(e),
                "total_task_time_ms": total_task_time_ms,
                "processing_time_ms": None,  # LLM call failed
            },
            "response_file": None,
        }
