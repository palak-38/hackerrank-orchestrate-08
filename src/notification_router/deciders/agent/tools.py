from agents import RunContextWrapper, function_tool

from notification_router.deciders.agent.context import NotificationContext


@function_tool
def search_history(
    ctx: RunContextWrapper[NotificationContext],
    query: str,
) -> str:
    """Search historical messages relevant to the current user.

    Args:
        query: Semantic search query describing the relevant topic.
    """

    summary, _ = ctx.context.evidence_tools.search_history(query)
    return summary


@function_tool
def find_messages_from_sender(
    ctx: RunContextWrapper[NotificationContext],
    scope: str,
) -> str:
    """Retrieve previous messages from the current message's sender.

    Args:
        scope: Must be "current_sender". The sender is determined
            from the authenticated execution context.
    """

    if scope != "current_sender":
        raise ValueError("Unsupported sender lookup scope")

    summary, _ = ctx.context.evidence_tools.find_messages_from_sender()
    return summary
