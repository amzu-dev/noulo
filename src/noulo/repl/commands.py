"""Slash-command registry: the single source for /help and tab completion."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Command:
    name: str
    args: str
    help: str
    group: str


COMMANDS: tuple[Command, ...] = (
    Command(
        "noul",
        "[proposition]",
        "Check whether a statement is true for each input you type",
        "Decide",
    ),
    Command("choice", "[question]", "Pick one of your options for each input", "Decide"),
    Command("score", "[question]", "Place each input on an ordered rubric (0-1)", "Decide"),
    Command("context", "", "Show the current mode, question, options or rubric", "Decide"),
    Command("proposition", "<text>", "Change the Noul statement", "Decide"),
    Command("question", "<text>", "Change the Choice/Score question", "Decide"),
    Command("options", "A=Billing, B=Sales", "Replace the Choice options", "Decide"),
    Command("rubric", "low, medium, high", "Replace the Score rubric (lowest first)", "Decide"),
    Command("good", "", "Tell noulo the last answer was right", "Teach"),
    Command("bad", "", "Tell noulo the last answer was wrong", "Teach"),
    Command("correct", "<answer>", "Give the right answer for the last input", "Teach"),
    Command("teach", "<file>", "Teach from a file of labelled examples (.jsonl)", "Teach"),
    Command("learning", "[on|off]", "Show or switch learning from inputs", "Teach"),
    Command("memory", "[clear]", "Show (or clear) remembered cases", "Teach"),
    Command("model", "[id|list]", "Choose a model: size, quantisation and RAM shown", "Configure"),
    Command(
        "config",
        "[show|get|set|unset]",
        "View or change settings (interactive editor without args)",
        "Configure",
    ),
    Command("status", "", "Service, model and learning status", "Service"),
    Command("start", "", "Start the background service", "Service"),
    Command("stop", "", "Stop the background service", "Service"),
    Command("restart", "", "Restart the service (applies config changes)", "Service"),
    Command("logs", "", "Show the service log", "Service"),
    Command("open", "", "Open the web frontend in your browser", "Service"),
    Command("json", "", "Toggle raw JSON output", "View"),
    Command("verbose", "", "Toggle probabilities and learning details", "View"),
    Command("clear", "", "Clear the screen", "View"),
    Command("help", "", "Show this help", "View"),
    Command("exit", "", "Leave noulo (Ctrl+D)", "View"),
)

ALIASES = {
    "quit": "exit",
    "q": "exit",
    "models": "model",
    "mode": "context",
    "?": "help",
    "settings": "config",
}

BY_NAME = {c.name: c for c in COMMANDS}


def resolve(name: str) -> str | None:
    name = ALIASES.get(name, name)
    return name if name in BY_NAME else None
