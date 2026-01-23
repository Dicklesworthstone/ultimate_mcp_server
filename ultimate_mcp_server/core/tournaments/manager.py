# --- core/tournaments/manager.py (Updates) ---
import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple  # Added Type

from pydantic import ValidationError

import ultimate_mcp_server.core.evaluation.evaluators  # Ensures evaluators are registered  # noqa: F401
from ultimate_mcp_server.core.evaluation.base import EVALUATOR_REGISTRY, Evaluator
from ultimate_mcp_server.core.models.tournament import (
    CreateTournamentInput,
    TournamentConfig,  # ModelConfig is nested in TournamentConfig from CreateTournamentInput
    TournamentData,
    TournamentRoundResult,
    TournamentStatus,
)
from ultimate_mcp_server.core.models.tournament import (
    ModelConfig as CoreModelConfig,  # Alias to avoid confusion
)
from ultimate_mcp_server.utils import get_logger

logger = get_logger("ultimate_mcp_server.tournaments.manager")

STORAGE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "storage"
TOURNAMENT_STORAGE_BASE = STORAGE_DIR / "tournaments"


class TournamentManager:
    def __init__(self):
        self.tournaments: Dict[str, TournamentData] = {}
        # --- NEW: Store instantiated evaluators per tournament ---
        self.tournament_evaluators: Dict[str, List[Evaluator]] = {}
        TOURNAMENT_STORAGE_BASE.mkdir(parents=True, exist_ok=True)
        logger.info(f"Tournament storage initialized at: {TOURNAMENT_STORAGE_BASE}")
        self._load_all_tournaments()

    def _instantiate_evaluators(self, tournament_id: str, config: TournamentConfig) -> bool:
        """Instantiates and stores evaluators for a tournament."""
        self.tournament_evaluators[tournament_id] = []
        for eval_config in config.evaluators:
            evaluator_cls = EVALUATOR_REGISTRY.get(eval_config.type)
            if not evaluator_cls:
                logger.error(
                    f"Unknown evaluator type '{eval_config.type}' for tournament {tournament_id}. Skipping."
                )
                # Optionally, fail tournament creation if a critical evaluator is missing
                continue
            try:
                self.tournament_evaluators[tournament_id].append(evaluator_cls(eval_config.params))
                logger.info(
                    f"Instantiated evaluator '{eval_config.type}' (ID: {eval_config.evaluator_id}) for tournament {tournament_id}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to instantiate evaluator '{eval_config.type}' (ID: {eval_config.evaluator_id}): {e}",
                    exc_info=True,
                )
                # Decide if this is a fatal error for the tournament
                return False  # Example: Fail if any evaluator instantiation fails
        return True

    def get_evaluators_for_tournament(self, tournament_id: str) -> List[Evaluator]:
        """Returns the list of instantiated evaluators for a given tournament."""
        return self.tournament_evaluators.get(tournament_id, [])

    def create_tournament(self, input_data: CreateTournamentInput) -> Optional[TournamentData]:
        try:
            logger.debug(
                f"Creating tournament with name: {input_data.name}, {len(input_data.model_configs)} model configs"
            )

            # Map input ModelConfig to core ModelConfig used in TournamentConfig
            core_model_configs = [
                CoreModelConfig(
                    model_id=mc.model_id,
                    diversity_count=mc.diversity_count,
                    temperature=mc.temperature,
                    max_tokens=mc.max_tokens,
                    system_prompt=mc.system_prompt,
                    seed=mc.seed,
                )
                for mc in input_data.model_configs
            ]

            tournament_cfg = TournamentConfig(
                name=input_data.name,
                prompt=input_data.prompt,
                models=core_model_configs,  # Use the mapped core_model_configs
                rounds=input_data.rounds,
                tournament_type=input_data.tournament_type,
                extraction_model_id=input_data.extraction_model_id,
                evaluators=input_data.evaluators,  # Pass evaluator configs
                max_retries_per_model_call=input_data.max_retries_per_model_call,
                retry_backoff_base_seconds=input_data.retry_backoff_base_seconds,
                max_concurrent_model_calls=input_data.max_concurrent_model_calls,
            )

            tournament = TournamentData(
                name=input_data.name,
                config=tournament_cfg,
                current_round=-1,  # Initialize current_round
                start_time=None,  # Will be set when execution starts
                end_time=None,
            )

            tournament.storage_path = str(
                self._get_storage_path(tournament.tournament_id, tournament.name)
            )  # Pass name for better paths

            # --- NEW: Instantiate evaluators ---
            if not self._instantiate_evaluators(tournament.tournament_id, tournament.config):
                logger.error(
                    f"Failed to instantiate one or more evaluators for tournament {tournament.name}. Creation aborted."
                )
                # Clean up if necessary, e.g., remove from self.tournament_evaluators
                if tournament.tournament_id in self.tournament_evaluators:
                    del self.tournament_evaluators[tournament.tournament_id]
                return None  # Or raise an error

            self.tournaments[tournament.tournament_id] = tournament
            self._save_tournament_state(tournament)
            logger.info(
                f"Tournament '{tournament.name}' (ID: {tournament.tournament_id}) created successfully."
            )
            return tournament
        except ValidationError as ve:
            logger.error(f"Tournament validation failed: {ve}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error creating tournament: {e}", exc_info=True)
            return None

    def get_tournament(
        self, tournament_id: str, force_reload: bool = False
    ) -> Optional[TournamentData]:
        logger.debug(f"Getting tournament {tournament_id} (force_reload={force_reload})")
        if not force_reload and tournament_id in self.tournaments:
            return self.tournaments[tournament_id]

        tournament = self._load_tournament_state(tournament_id)
        if tournament:
            # --- NEW: Ensure evaluators are loaded/re-instantiated if not present ---
            if tournament_id not in self.tournament_evaluators:
                logger.info(
                    f"Evaluators for tournament {tournament_id} not in memory, re-instantiating from config."
                )
                if not self._instantiate_evaluators(tournament_id, tournament.config):
                    logger.error(
                        f"Failed to re-instantiate evaluators for loaded tournament {tournament_id}. Evaluation might fail."
                    )
            self.tournaments[tournament_id] = tournament  # Update cache
        return tournament

    def _save_tournament_state(self, tournament: TournamentData):
        if not tournament.storage_path:
            logger.error(
                f"Cannot save state for tournament {tournament.tournament_id}: storage_path not set."
            )
            return

        state_file = Path(tournament.storage_path) / "tournament_state.json"
        try:
            state_file.parent.mkdir(parents=True, exist_ok=True)
            # Pydantic's model_dump_json handles datetime to ISO string conversion
            json_data = tournament.model_dump_json(indent=2)
            with open(state_file, "w", encoding="utf-8") as f:
                f.write(json_data)
            logger.debug(f"Saved state for tournament {tournament.tournament_id} to {state_file}")
        except IOError as e:
            logger.error(f"Failed to save state for tournament {tournament.tournament_id}: {e}")
        except Exception as e:  # Catch other potential errors from model_dump_json
            logger.error(
                f"Error serializing tournament state for {tournament.tournament_id}: {e}",
                exc_info=True,
            )

    def _load_tournament_state(self, tournament_id: str) -> Optional[TournamentData]:
        # Try finding by explicit ID first (common case for direct access)
        # The storage path might be complex now, so scan might be more reliable if ID is the only input

        # Robust scan: iterate through all subdirectories of TOURNAMENT_STORAGE_BASE
        if TOURNAMENT_STORAGE_BASE.exists():
            for potential_tournament_dir in TOURNAMENT_STORAGE_BASE.iterdir():
                if potential_tournament_dir.is_dir():
                    state_file = potential_tournament_dir / "tournament_state.json"
                    if state_file.exists():
                        try:
                            with open(state_file, "r", encoding="utf-8") as f:
                                data = json.load(f)
                                if data.get("tournament_id") == tournament_id:
                                    # Use Pydantic for robust parsing and type conversion
                                    parsed_tournament = TournamentData.model_validate(data)
                                    logger.debug(
                                        f"Loaded state for tournament {tournament_id} from {state_file}"
                                    )
                                    return parsed_tournament
                        except (IOError, json.JSONDecodeError, ValidationError) as e:
                            logger.warning(
                                f"Failed to load or validate state from {state_file} for tournament ID {tournament_id}: {e}"
                            )
                            # Don't return, continue scanning
                        except Exception as e:  # Catch any other unexpected error
                            logger.error(
                                f"Unexpected error loading state from {state_file}: {e}",
                                exc_info=True,
                            )

        logger.debug(f"Tournament {tournament_id} not found in any storage location during scan.")
        return None

    def _load_all_tournaments(self):
        logger.info(f"Scanning {TOURNAMENT_STORAGE_BASE} for existing tournaments...")
        count = 0
        if not TOURNAMENT_STORAGE_BASE.exists():
            logger.warning("Tournament storage directory does not exist. No tournaments loaded.")
            return

        for item in TOURNAMENT_STORAGE_BASE.iterdir():
            if item.is_dir():
                # Attempt to load tournament_state.json from this directory
                state_file = item / "tournament_state.json"
                if state_file.exists():
                    try:
                        with open(state_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        tournament_id_from_file = data.get("tournament_id")
                        if not tournament_id_from_file:
                            logger.warning(
                                f"Skipping directory {item.name}, tournament_state.json missing 'tournament_id'."
                            )
                            continue

                        if (
                            tournament_id_from_file not in self.tournaments
                        ):  # Avoid reloading if already cached by some other means
                            # Use the get_tournament method which handles re-instantiating evaluators
                            loaded_tournament = self.get_tournament(
                                tournament_id_from_file, force_reload=True
                            )
                            if loaded_tournament:
                                count += 1
                                logger.debug(
                                    f"Loaded tournament '{loaded_tournament.name}' (ID: {loaded_tournament.tournament_id}) from {item.name}"
                                )
                            else:
                                logger.warning(
                                    f"Failed to fully load tournament from directory: {item.name} (ID in file: {tournament_id_from_file})"
                                )
                    except (IOError, json.JSONDecodeError, ValidationError) as e:
                        logger.warning(f"Error loading tournament from directory {item.name}: {e}")
                    except Exception as e:
                        logger.error(
                            f"Unexpected error loading tournament from {item.name}: {e}",
                            exc_info=True,
                        )
        logger.info(f"Finished scan. Loaded {count} existing tournaments into manager.")

    def start_tournament_execution(self, tournament_id: str) -> bool:
        logger.debug(f"Attempting to start tournament execution for {tournament_id}")
        tournament = self.get_tournament(tournament_id)  # Ensures evaluators are loaded
        if not tournament:
            logger.error(f"Cannot start execution: Tournament {tournament_id} not found.")
            return False

        if tournament.status not in [TournamentStatus.PENDING, TournamentStatus.CREATED]:
            logger.warning(
                f"Tournament {tournament_id} is not in a runnable state ({tournament.status}). Cannot start."
            )
            return False

        tournament.status = TournamentStatus.RUNNING  # Or QUEUED if worker mode is implemented
        tournament.start_time = datetime.now(timezone.utc)
        tournament.current_round = 0  # Explicitly set to 0 when starting
        # Ensure rounds_results is initialized if empty
        if not tournament.rounds_results:
            tournament.rounds_results = [
                TournamentRoundResult(round_num=i) for i in range(tournament.config.rounds)
            ]

        self._save_tournament_state(tournament)
        logger.info(
            f"Tournament {tournament_id} status set to {tournament.status}, ready for async execution."
        )

        try:
            from ultimate_mcp_server.core.tournaments.tasks import (
                run_tournament_async,  # Local import
            )

            asyncio.create_task(run_tournament_async(tournament_id))
            logger.info(f"Asyncio task created for tournament {tournament_id}.")
            return True
        except Exception as e:
            logger.error(
                f"Error creating asyncio task for tournament {tournament_id}: {e}", exc_info=True
            )
            tournament.status = TournamentStatus.FAILED
            tournament.error_message = f"Failed during asyncio task creation: {str(e)}"
            tournament.end_time = datetime.now(timezone.utc)
            self._save_tournament_state(tournament)
            return False

    async def cancel_tournament(
        self, tournament_id: str
    ) -> Tuple[bool, str, TournamentStatus]:  # Return final status
        """Attempts to cancel a tournament. Returns success, message, and final status."""
        tournament = self.get_tournament(tournament_id, force_reload=True)
        if not tournament:
            logger.warning(f"Cannot cancel non-existent tournament {tournament_id}")
            # Use FAILED or a specific status for "not found" if added to enum,
            # or rely on the tool layer to raise 404. For manager, FAILED can represent this.
            return False, "Tournament not found.", TournamentStatus.FAILED

        current_status = tournament.status
        final_status = current_status  # Default to current status if no change
        message = ""

        if current_status == TournamentStatus.RUNNING or current_status == TournamentStatus.QUEUED:
            logger.info(
                f"Attempting to cancel tournament {tournament_id} (status: {current_status})..."
            )
            tournament.status = TournamentStatus.CANCELLED
            tournament.error_message = (
                tournament.error_message or "Tournament cancelled by user request."
            )
            tournament.end_time = datetime.now(timezone.utc)
            final_status = TournamentStatus.CANCELLED
            message = "Cancellation requested. Tournament status set to CANCELLED."
            self._save_tournament_state(tournament)
            logger.info(f"Tournament {tournament_id} status set to CANCELLED.")
            # The background task needs to observe this status.
            return True, message, final_status
        elif current_status in [
            TournamentStatus.COMPLETED,
            TournamentStatus.FAILED,
            TournamentStatus.CANCELLED,
        ]:
            message = f"Tournament {tournament_id} is already finished or cancelled (Status: {current_status})."
            logger.warning(message)
            return False, message, final_status
        elif (
            current_status == TournamentStatus.PENDING or current_status == TournamentStatus.CREATED
        ):
            tournament.status = TournamentStatus.CANCELLED
            tournament.error_message = "Tournament cancelled before starting."
            tournament.end_time = datetime.now(timezone.utc)
            final_status = TournamentStatus.CANCELLED
            message = "Pending/Created tournament cancelled successfully."
            self._save_tournament_state(tournament)
            logger.info(f"Pending/Created tournament {tournament_id} cancelled.")
            return True, message, final_status
        else:
            # Should not happen, but handle unknown state
            message = f"Tournament {tournament_id} is in an unexpected state ({current_status}). Cannot determine cancellation action."
            logger.error(message)
            return False, message, current_status

    def list_tournaments(self) -> List[Dict[str, Any]]:
        # Ensure cache is up-to-date if new tournaments might have been added externally (less likely with file storage)
        # self._load_all_tournaments() # Consider if this is too expensive for every list call

        basic_list = []
        for t_data in self.tournaments.values():
            basic_list.append(
                {
                    "tournament_id": t_data.tournament_id,
                    "name": t_data.name,
                    "tournament_type": t_data.config.tournament_type,
                    "status": t_data.status,
                    "current_round": t_data.current_round,
                    "total_rounds": t_data.config.rounds,
                    "created_at": t_data.created_at.isoformat()
                    if t_data.created_at
                    else None,  # Ensure ISO format
                    "updated_at": t_data.updated_at.isoformat() if t_data.updated_at else None,
                    "start_time": t_data.start_time.isoformat() if t_data.start_time else None,
                    "end_time": t_data.end_time.isoformat() if t_data.end_time else None,
                }
            )
        basic_list.sort(
            key=lambda x: x["created_at"] or "", reverse=True
        )  # Handle None created_at for sorting
        return basic_list

    def _get_storage_path(self, tournament_id: str, tournament_name: str) -> Path:
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Sanitize tournament name for use in path
        safe_name = re.sub(r"[^\w\s-]", "", tournament_name).strip().replace(" ", "_")
        safe_name = re.sub(
            r"[-\s]+", "-", safe_name
        )  # Replace multiple spaces/hyphens with single hyphen
        safe_name = safe_name[:50]  # Limit length

        # Use first 8 chars of UUID for brevity if name is too generic or empty
        uuid_suffix = tournament_id.split("-")[0]

        folder_name = (
            f"{timestamp_str}_{safe_name}_{uuid_suffix}"
            if safe_name
            else f"{timestamp_str}_{uuid_suffix}"
        )

        path = TOURNAMENT_STORAGE_BASE / folder_name
        path.mkdir(parents=True, exist_ok=True)  # Ensure directory is created
        return path


tournament_manager = TournamentManager()
