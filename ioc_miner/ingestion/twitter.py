"""
Twitter/X ingestor via tweepy (API v2).

Credentials via env vars:
  TW_BEARER_TOKEN   — required, Twitter API v2 Bearer Token
  TW_API_KEY        — optional, for user-context endpoints
  TW_API_SECRET     — optional
  TW_ACCESS_TOKEN   — optional
  TW_ACCESS_SECRET  — optional

Usage examples:
  TwitterIngestor().ingest("@vxunderground")        # user timeline
  TwitterIngestor(mode="search").ingest("#threatintel malware IOC")  # search
  TwitterIngestor(mode="list").ingest("1234567890")  # list ID

Note on snscrape: broken since 2023 Twitter/X API lockdown — not used.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from .base import BaseIngestor

_DEFAULT_MAX_RESULTS = 100
_DEFAULT_LOOKBACK_DAYS = 7


class TwitterIngestor(BaseIngestor):
    """
    Fetches tweets from Twitter/X API v2.

    Args:
        mode: "user" (timeline), "search" (recent search), or "list"
        max_results: max tweets to return (10–100 per page, auto-paginated)
        lookback_days: only fetch tweets from the last N days
    """

    source_type = "twitter"

    def __init__(
        self,
        mode: str = "user",
        max_results: int = _DEFAULT_MAX_RESULTS,
        lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
    ):
        if mode not in ("user", "search", "list"):
            raise ValueError(f"mode must be 'user', 'search', or 'list', got {mode!r}")
        self.mode = mode
        self.max_results = max_results
        self.lookback_days = lookback_days

    def ingest(self, source: str) -> str:
        """
        Args:
            source:
              mode=user   → Twitter handle, e.g. '@vxunderground' or 'vxunderground'
              mode=search → search query, e.g. '#threatintel CVE'
              mode=list   → list ID as string, e.g. '1234567890'
        Returns:
            All tweet texts joined by double newlines, chronological order.
        """
        try:
            import tweepy
        except ImportError:
            raise ImportError("tweepy is required: pip install 'ioc-miner[twitter]'")

        bearer = os.environ.get("TW_BEARER_TOKEN")
        if not bearer:
            raise EnvironmentError("Set TW_BEARER_TOKEN environment variable")

        client = tweepy.Client(
            bearer_token=bearer,
            api_key=os.environ.get("TW_API_KEY"),
            api_secret_key=os.environ.get("TW_API_SECRET"),
            access_token=os.environ.get("TW_ACCESS_TOKEN"),
            access_token_secret=os.environ.get("TW_ACCESS_SECRET"),
            wait_on_rate_limit=True,
        )

        start_time = datetime.now(timezone.utc) - timedelta(days=self.lookback_days)

        if self.mode == "user":
            tweets = self._fetch_user_timeline(client, source, start_time)
        elif self.mode == "search":
            tweets = self._fetch_search(client, source, start_time)
        else:
            tweets = self._fetch_list(client, source, start_time)

        return "\n\n".join(reversed(tweets))  # chronological

    # ── per-mode helpers ──────────────────────────────────────────────────────

    def _fetch_user_timeline(self, client, handle: str, start_time: datetime) -> list[str]:
        import tweepy

        handle = handle.lstrip("@")
        user_resp = client.get_user(username=handle)
        if not user_resp.data:
            raise ValueError(f"Twitter user not found: @{handle}")
        user_id = user_resp.data.id

        texts: list[str] = []
        paginator = tweepy.Paginator(
            client.get_users_tweets,
            id=user_id,
            start_time=start_time,
            tweet_fields=["text", "created_at"],
            exclude=["retweets", "replies"],
            max_results=min(self.max_results, 100),
        )
        for tweet in paginator.flatten(limit=self.max_results):
            if tweet.text:
                texts.append(tweet.text)
        return texts

    def _fetch_search(self, client, query: str, start_time: datetime) -> list[str]:
        import tweepy

        # Exclude retweets automatically
        full_query = f"({query}) -is:retweet lang:en"

        texts: list[str] = []
        paginator = tweepy.Paginator(
            client.search_recent_tweets,
            query=full_query,
            start_time=start_time,
            tweet_fields=["text", "created_at"],
            max_results=min(self.max_results, 100),
        )
        for tweet in paginator.flatten(limit=self.max_results):
            if tweet.text:
                texts.append(tweet.text)
        return texts

    def _fetch_list(self, client, list_id: str, start_time: datetime) -> list[str]:
        import tweepy

        texts: list[str] = []
        paginator = tweepy.Paginator(
            client.get_list_tweets,
            id=list_id,
            tweet_fields=["text", "created_at"],
            max_results=min(self.max_results, 100),
        )
        for tweet in paginator.flatten(limit=self.max_results):
            if tweet.text:
                texts.append(tweet.text)
        return texts
