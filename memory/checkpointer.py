import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from config.settings import settings


def get_sqlite_checkpointer() -> SqliteSaver:
    """Returns a persistent SQLite checkpointer for LangGraph thread memory."""
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(settings.sqlite_db_path), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver
