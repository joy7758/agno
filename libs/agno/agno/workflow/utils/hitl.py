"""Helper classes and functions for workflow continue execution.

This module contains shared utilities used by the continue_run methods
(sync/async, streaming/non-streaming) to reduce code duplication.
"""

from typing import TYPE_CHECKING, Dict, List, Union, cast

from agno.run.base import RunStatus
from agno.workflow.types import StepOutput

if TYPE_CHECKING:
    from agno.media import Audio, File, Image, Video
    from agno.run.workflow import WorkflowRunOutput
    from agno.workflow.types import WorkflowExecutionInput


class ContinueExecutionState:
    """State container for continue execution methods.

    This class encapsulates the shared state used across all continue_execute variants
    (sync/async, streaming/non-streaming) to reduce code duplication.
    """

    def __init__(
        self,
        workflow_run_response: "WorkflowRunOutput",
        execution_input: "WorkflowExecutionInput",
    ):
        # Restore previous step outputs from step_results
        self.collected_step_outputs: List[Union["StepOutput", List["StepOutput"]]] = list(
            workflow_run_response.step_results or []
        )
        self.previous_step_outputs: Dict[str, "StepOutput"] = {}
        for step_output in self.collected_step_outputs:
            if isinstance(step_output, StepOutput) and step_output.step_name:
                self.previous_step_outputs[step_output.step_name] = step_output

        # Initialize media lists
        self.shared_images: List["Image"] = execution_input.images or []
        self.output_images: List["Image"] = (execution_input.images or []).copy()
        self.shared_videos: List["Video"] = execution_input.videos or []
        self.output_videos: List["Video"] = (execution_input.videos or []).copy()
        self.shared_audio: List["Audio"] = execution_input.audio or []
        self.output_audio: List["Audio"] = (execution_input.audio or []).copy()
        self.shared_files: List["File"] = execution_input.files or []
        self.output_files: List["File"] = (execution_input.files or []).copy()

        # Restore shared media from previous steps
        for step_output in self.collected_step_outputs:
            if isinstance(step_output, StepOutput):
                self.shared_images.extend(step_output.images or [])
                self.shared_videos.extend(step_output.videos or [])
                self.shared_audio.extend(step_output.audio or [])
                self.shared_files.extend(step_output.files or [])
                self.output_images.extend(step_output.images or [])
                self.output_videos.extend(step_output.videos or [])
                self.output_audio.extend(step_output.audio or [])
                self.output_files.extend(step_output.files or [])

    def extend_media_from_step(self, step_output: "StepOutput") -> None:
        """Extend shared and output media lists from a step output."""
        self.shared_images.extend(step_output.images or [])
        self.shared_videos.extend(step_output.videos or [])
        self.shared_audio.extend(step_output.audio or [])
        self.shared_files.extend(step_output.files or [])
        self.output_images.extend(step_output.images or [])
        self.output_videos.extend(step_output.videos or [])
        self.output_audio.extend(step_output.audio or [])
        self.output_files.extend(step_output.files or [])

    def add_step_output(self, step_name: str, step_output: "StepOutput") -> None:
        """Add a step output to tracking collections and extend media."""
        self.previous_step_outputs[step_name] = step_output
        self.collected_step_outputs.append(step_output)
        self.extend_media_from_step(step_output)


def finalize_workflow_completion(
    workflow_run_response: "WorkflowRunOutput",
    state: ContinueExecutionState,
) -> None:
    """Finalize workflow completion by updating metrics and status.

    This helper consolidates the common completion logic used across all
    continue_execute variants.

    Args:
        workflow_run_response: The workflow run output to finalize.
        state: The execution state containing collected outputs and media.
    """
    if state.collected_step_outputs:
        if workflow_run_response.metrics:
            workflow_run_response.metrics.stop_timer()

        # Extract final content from last step output
        last_output = cast(StepOutput, state.collected_step_outputs[-1])

        if getattr(last_output, "steps", None):
            _cur = last_output
            while getattr(_cur, "steps", None):
                _steps = _cur.steps or []
                if not _steps:
                    break
                _cur = _steps[-1]
            workflow_run_response.content = _cur.content
        else:
            workflow_run_response.content = last_output.content
    else:
        workflow_run_response.content = "No steps executed"

    workflow_run_response.step_results = state.collected_step_outputs
    workflow_run_response.images = state.output_images
    workflow_run_response.videos = state.output_videos
    workflow_run_response.audio = state.output_audio
    workflow_run_response.status = RunStatus.completed
    workflow_run_response._paused_step_index = None
