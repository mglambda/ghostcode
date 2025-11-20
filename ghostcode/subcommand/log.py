# ghostcode.subcommand.log
from typing import *
from pydantic import BaseModel, Field
import os
import sys
from .. import types
from ..types import CommandOutput
from .. import git
from ..program import CommandInterface, Program
import logging
logger = logging.getLogger("ghostcode.subcommand.log")

class LogCommand(CommandInterface):
    """Manages and displays interaction history."""

    interaction_identifier: Optional[str] = Field(
        default=None,
        description="Optional unique ID or tag of a specific interaction to display in detail.",
    )

    all_branches: bool = Field(
        default=False, description="Do not filter any interactions based on branches."
    )

    def _overview_header(self, interaction: types.InteractionHistory) -> str:
        num_turns = len(interaction.contents)
        turns_str = f"turns: {num_turns}\n"
        tag_str = f"tag: {interaction.tag}\n" if interaction.tag else ""
        branch_str = (
            f"branch: {interaction.branch_name}\n"
            if interaction.branch_name is not None
            else ""
        )
        git_str = (
            f"commits affected: {", ".join(commits)}\n"
            if (commits := interaction.get_affected_git_commits()) != []
            else ""
        )
        title_str = f"title: {interaction.title}\n" if interaction.title else ""
        if (timestamps := interaction.timestamps()) is not None:
            time_str = (
                f"date started: {timestamps[0]}\nlast interaction: {timestamps[1]}\n"
            )
        else:
            time_str = ""
        return f"""interaction {interaction.unique_id}
{tag_str}{branch_str}{git_str}{turns_str}{time_str}{title_str}"""

    def _overview_show_interaction(self, interaction: types.InteractionHistory) -> str:
        num_turns = len(interaction.contents)
        if interaction.empty():
            first_msg = ""
        else:
            match interaction.contents[0]:
                case types.UserInteractionHistoryItem() as user_item:
                    first_msg = user_item.prompt.strip()
                case _:
                    first_msg = ""

            if first_msg:
                limit = 120
                if len(first_msg) >= limit:
                    first_msg = first_msg[:limit].strip() + "..."

        return f"""{self._overview_header(interaction)}
{first_msg}
"""

    def run(self, prog: Program) -> CommandOutput:
        result = CommandOutput()
        if not prog.project_root or not prog.project:
            logger.error("Not a ghostcode project. Run 'ghostcode init' first.")
            sys.exit(1)

        project = prog.project

        if self.interaction_identifier:
            # Detailed view
            target_id = self.interaction_identifier

            # 1. Try to find by unique_id (should be unique)
            found_by_id = [i for i in project.interactions if i.unique_id == target_id]
            if len(found_by_id) == 1:
                interaction = found_by_id[0]
                result.print(self._overview_header(interaction))
                result.print(interaction.show())
                return result
            elif len(found_by_id) > 1:
                # This should ideally not happen for unique_id, but handle defensively
                result.print(
                    f"CRITICAL ERROR: Multiple interactions found with the same unique ID '{target_id}'. This indicates data corruption."
                )
                return result

            # 2. If not found by unique_id, try to find by tag
            found_by_tag = [i for i in project.interactions if i.tag == target_id]
            if len(found_by_tag) == 1:
                interaction = found_by_tag[0]
                result.print(self._overview_header(interaction))
                result.print(interaction.show())
                return result
            elif len(found_by_tag) > 1:
                result.print(
                    f"Multiple interactions found with tag '{target_id}'. Please use the full unique ID for detailed view:"
                )
                for interaction in found_by_tag:
                    result.print(self._overview_header(interaction))
                return result
            else:
                result.print(f"No interaction found with tag or ID '{target_id}'.")
                return result
        else:
            # Overview mode
            if not project.interactions:
                result.print("No past interactions found.")
            else:
                # FIXME: actually sort by date
                # note: sorting by date is slightly more complicated, e.g.: what date? date started or date last interacted? for now this will do
                branch_gr = git.get_current_branch(prog.project_root)
                current_branch = branch_gr.value if branch_gr is not None else ""
                num_filtered = 0
                for interaction in reversed(project.interactions):
                    # check branches, unless all is requested
                    if not self.all_branches:
                        if interaction.branch_name != current_branch:
                            num_filtered += 1
                            continue

                    result.print(self._overview_show_interaction(interaction))

            if num_filtered > 0:
                result.print(
                    f"{num_filtered} interaction histories have been filtered."
                )
                if prog.user_config.newbie:
                    result.print(
                        f"Hint: By default, only interactions that were started on your current git branch are shown. Use --all-branches to disable this behaviour, or checkout a different branch."
                    )
        return result

