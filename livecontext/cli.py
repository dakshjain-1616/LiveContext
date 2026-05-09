"""LiveContext CLI using Click."""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from livecontext.server.db import Database, get_db
from livecontext.server.models import SessionInfo

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="livecontext")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx: click.Context, verbose: bool):
    """LiveContext - Real-time streaming context window monitor for LLM agents."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.option("--host", "-h", default="0.0.0.0", help="Host to bind to")
@click.option("--port", "-p", default=8000, type=int, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
@click.option("--workers", "-w", default=1, type=int, help="Number of workers")
def serve(host: str, port: int, reload: bool, workers: int):
    """Start the LiveContext server."""
    import uvicorn
    
    console.print(Panel.fit(
        f"[bold cyan]LiveContext Server[/bold cyan]\n"
        f"Starting on [green]{host}:{port}[/green]\n"
        f"Reload: {'[green]enabled[/green]' if reload else '[dim]disabled[/dim]'}\n"
        f"Workers: [yellow]{workers}[/yellow]",
        title="🚀 Server",
        border_style="cyan"
    ))
    
    uvicorn.run(
        "livecontext.server.app:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
        log_level="info"
    )


@cli.command()
@click.option("--db-path", "-d", default="data/livecontext.db", help="Database path")
def status(db_path: str):
    """Show server status and database statistics."""
    db = Database(db_path)
    stats = db.get_stats()
    
    # Create status table
    table = Table(title="LiveContext Status", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Database Path", db_path)
    table.add_row("Sessions", str(stats.get("sessions_count", 0)))
    table.add_row("Messages", str(stats.get("messages_count", 0)))
    table.add_row("Snapshots", str(stats.get("snapshots_count", 0)))
    table.add_row("Evictions", str(stats.get("evictions_count", 0)))
    table.add_row("Cached Embeddings", str(stats.get("embeddings_cache_count", 0)))
    table.add_row("Cache Hits", str(stats.get("embedding_cache_hits", 0)))
    
    console.print(table)
    
    # Show recent sessions
    sessions = db.list_sessions()
    if sessions:
        session_table = Table(title="Recent Sessions", show_header=True)
        session_table.add_column("ID", style="dim")
        session_table.add_column("Model", style="cyan")
        session_table.add_column("Provider", style="blue")
        session_table.add_column("Messages", justify="right")
        session_table.add_column("Status", style="green")
        
        for session in sessions[:10]:
            status_text = "[green]active[/green]" if session.is_active else "[dim]inactive[/dim]"
            session_table.add_row(
                session.id[:8] + "...",
                session.model_name,
                session.provider,
                str(session.message_count),
                status_text
            )
        
        console.print(session_table)
    else:
        console.print("[dim]No sessions found.[/dim]")


@cli.command()
@click.argument("session_id", required=False)
@click.option("--limit", "-l", default=10, type=int, help="Number of snapshots to show")
def snapshots(session_id: Optional[str], limit: int):
    """List snapshots for a session."""
    db = get_db()
    
    if not session_id:
        # List all sessions first
        sessions = db.list_sessions()
        if not sessions:
            console.print("[yellow]No sessions found. Create a session first.[/yellow]")
            return
        
        session_table = Table(title="Available Sessions")
        session_table.add_column("#", justify="right")
        session_table.add_column("ID", style="cyan")
        session_table.add_column("Model")
        session_table.add_column("Messages", justify="right")
        
        for i, session in enumerate(sessions[:10], 1):
            session_table.add_row(
                str(i),
                session.id[:16] + "...",
                session.model_name,
                str(session.message_count)
            )
        
        console.print(session_table)
        return
    
    # Show snapshots for specific session
    snapshots_list = db.get_snapshots(session_id, limit)
    
    if not snapshots_list:
        console.print(f"[yellow]No snapshots found for session {session_id[:8]}...[/yellow]")
        return
    
    snap_table = Table(title=f"Snapshots for {session_id[:16]}...")
    snap_table.add_column("Time", style="cyan")
    snap_table.add_column("Tokens", justify="right")
    snap_table.add_column("Utilization", justify="right")
    snap_table.add_column("Messages", justify="right")
    snap_table.add_column("Evictions", justify="right")
    
    for snap in snapshots_list:
        util_color = "green" if snap.utilization_percent < 50 else "yellow" if snap.utilization_percent < 75 else "red"
        snap_table.add_row(
            snap.timestamp.strftime("%H:%M:%S"),
            str(snap.total_tokens),
            f"[{util_color}]{snap.utilization_percent:.1f}%[/{util_color}]",
            str(len(snap.messages)),
            str(len(snap.evictions))
        )
    
    console.print(snap_table)


@cli.command()
@click.argument("session_id")
def session(session_id: str):
    """Show details for a specific session."""
    db = get_db()
    session = db.get_session(session_id)
    
    if not session:
        console.print(f"[red]Session not found: {session_id}[/red]")
        return
    
    # Session details panel
    details = Text()
    details.append(f"ID: ", style="bold cyan")
    details.append(f"{session.id}\n")
    details.append(f"Model: ", style="bold cyan")
    details.append(f"{session.model_name}\n")
    details.append(f"Provider: ", style="bold cyan")
    details.append(f"{session.provider}\n")
    details.append(f"Max Tokens: ", style="bold cyan")
    details.append(f"{session.max_tokens:,}\n")
    details.append(f"Messages: ", style="bold cyan")
    details.append(f"{session.message_count}\n")
    details.append(f"Evictions: ", style="bold cyan")
    details.append(f"{session.total_evictions}\n")
    details.append(f"Status: ", style="bold cyan")
    details.append("Active" if session.is_active else "Inactive", style="green" if session.is_active else "dim")
    
    console.print(Panel(details, title="Session Details", border_style="cyan"))
    
    # Show recent messages
    messages = db.get_messages(session_id, limit=20)
    if messages:
        msg_table = Table(title="Recent Messages")
        msg_table.add_column("Role", style="cyan")
        msg_table.add_column("Content")
        msg_table.add_column("Tokens", justify="right")
        
        for msg in messages[-10:]:
            content = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
            role_color = {
                "system": "violet",
                "user": "green",
                "assistant": "blue",
                "tool": "yellow"
            }.get(msg.role.value, "white")
            
            msg_table.add_row(
                f"[{role_color}]{msg.role.value}[/{role_color}]",
                content,
                str(msg.token_count)
            )
        
        console.print(msg_table)


@cli.command()
@click.option("--host", "-h", default="localhost", help="Server host")
@click.option("--port", "-p", default=8000, type=int, help="Server port")
def dashboard(host: str, port: int):
    """Open the LiveContext dashboard in browser."""
    import webbrowser
    
    url = f"http://{host}:{port}"
    console.print(f"[cyan]Opening dashboard at {url}...[/cyan]")
    webbrowser.open(url)


@cli.command()
@click.option("--older-than", "-d", type=int, help="Clear entries older than N days")
def clear_cache(older_than: Optional[int]):
    """Clear the embedding cache."""
    db = get_db()
    count = db.clear_cache(older_than_days=older_than)
    
    if older_than:
        console.print(f"[green]Cleared {count} cache entries older than {older_than} days[/green]")
    else:
        console.print(f"[green]Cleared all {count} cache entries[/green]")


@cli.command()
@click.option("--model", "-m", default="gpt-4", help="Model name")
@click.option("--provider", "-p", default="openai", help="Provider (openai, anthropic, ollama)")
@click.option("--max-tokens", "-t", default=4096, type=int, help="Max tokens")
def create_session(model: str, provider: str, max_tokens: int):
    """Create a new monitoring session."""
    db = get_db()
    
    session = SessionInfo(
        model_name=model,
        provider=provider,
        max_tokens=max_tokens
    )
    
    db.create_session(session)
    
    console.print(Panel.fit(
        f"[bold green]Session Created[/bold green]\n\n"
        f"ID: [cyan]{session.id}[/cyan]\n"
        f"Model: [blue]{model}[/blue]\n"
        f"Provider: [blue]{provider}[/blue]\n"
        f"Max Tokens: [yellow]{max_tokens:,}[/yellow]",
        title="✅ Success",
        border_style="green"
    ))
    
    console.print(f"\n[dim]Use this ID with: livecontext session {session.id}[/dim]")


@cli.command()
@click.argument("session_id")
def delete_session(session_id: str):
    """Delete a session."""
    db = get_db()
    
    success = db.update_session(session_id, is_active=False)
    
    if success:
        console.print(f"[green]Session {session_id[:16]}... deleted successfully[/green]")
    else:
        console.print(f"[red]Session not found: {session_id}[/red]")


def main():
    """Entry point for CLI."""
    cli()


if __name__ == "__main__":
    main()
