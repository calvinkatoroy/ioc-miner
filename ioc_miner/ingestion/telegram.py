"""
Telegram channel ingestor via Telethon.

Usage requires a Telegram API id/hash from https://my.telegram.org/apps
Set via env vars: TG_API_ID, TG_API_HASH, TG_SESSION (optional session name)
"""

from __future__ import annotations

import asyncio
import os

from .base import BaseIngestor


class TelegramIngestor(BaseIngestor):
    """
    Fetches recent messages from a public Telegram channel.

    Args:
        limit: Max number of messages to fetch (default 200)
        min_date: Only fetch messages after this date (optional)
    """

    source_type = "telegram"

    def __init__(self, limit: int = 200):
        self.limit = limit

    def ingest(self, source: str) -> str:
        """
        Args:
            source: Channel username or invite link, e.g. '@threatintel_channel'
        """
        return asyncio.run(self._fetch(source))

    async def _fetch(self, channel: str) -> str:
        try:
            from telethon import TelegramClient
            from telethon.tl.types import MessageMediaDocument, MessageMediaPhoto
        except ImportError:
            raise ImportError("telethon is required: pip install 'ioc-miner[telegram]'")

        api_id = os.environ.get("TG_API_ID")
        api_hash = os.environ.get("TG_API_HASH")
        if not api_id or not api_hash:
            raise EnvironmentError("Set TG_API_ID and TG_API_HASH environment variables")

        session_name = os.environ.get("TG_SESSION", "ioc_miner_session")
        messages: list[str] = []

        async with TelegramClient(session_name, int(api_id), api_hash) as client:
            async for msg in client.iter_messages(channel, limit=self.limit):
                if msg.text:
                    messages.append(msg.text)

        return "\n\n".join(reversed(messages))  # chronological order
