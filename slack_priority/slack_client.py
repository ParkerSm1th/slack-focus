from __future__ import annotations

import json
import threading
import time
from collections import Counter
from dataclasses import dataclass
from typing import Callable

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_sdk.socket_mode import SocketModeClient
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse

from .config import Settings, score_to_level, should_notify
from .model import MODEL_VERSION, UrgencyScorer
from .notify import send_notification, slack_channel_url
from .storage import Store


CONVERSATION_TYPES = "public_channel,private_channel,im,mpim"
IMPORTABLE_SUBTYPES = {None, "bot_message", "thread_broadcast"}


@dataclass(frozen=True)
class ParsedMessage:
    message_id: str
    channel_id: str
    user_id: str
    text: str
    ts: str
    raw: dict


def parse_message_event(event: dict, *, team_id: str | None = None, self_user_id: str | None = None) -> ParsedMessage | None:
    parsed, _ = classify_message_event(event, team_id=team_id, self_user_id=self_user_id)
    return parsed


def classify_message_event(
    event: dict,
    *,
    team_id: str | None = None,
    self_user_id: str | None = None,
    fallback_channel_id: str | None = None,
) -> tuple[ParsedMessage | None, str]:
    if event.get("type") != "message":
        return None, "not_message"
    subtype = event.get("subtype")
    if subtype == "message_changed":
        changed_message = event.get("message")
        if not isinstance(changed_message, dict):
            return None, "changed_missing_message"

        nested_event = dict(changed_message)
        nested_event.setdefault("type", "message")
        nested_event.setdefault("channel", event.get("channel") or fallback_channel_id)
        parsed, reason = classify_message_event(
            nested_event,
            team_id=team_id,
            self_user_id=self_user_id,
            fallback_channel_id=fallback_channel_id or event.get("channel"),
        )
        if not parsed:
            return None, f"changed:{reason}"
        return (
            ParsedMessage(
                message_id=parsed.message_id,
                channel_id=parsed.channel_id,
                user_id=parsed.user_id,
                text=parsed.text,
                ts=parsed.ts,
                raw=event,
            ),
            "accepted_changed",
        )

    if subtype not in IMPORTABLE_SUBTYPES:
        return None, f"subtype:{event.get('subtype')}"

    text = str(event.get("text") or "").strip()
    channel_id = str(event.get("channel") or fallback_channel_id or "")
    user_id = str(
        event.get("user")
        or event.get("bot_id")
        or event.get("app_id")
        or event.get("username")
        or ""
    )
    ts = str(event.get("ts") or "")

    if not text:
        return None, "empty_text"
    if not channel_id:
        return None, "missing_channel"
    if not user_id:
        return None, "missing_user_or_bot"
    if not ts:
        return None, "missing_ts"
    if self_user_id and user_id == self_user_id:
        return None, "self_message"

    prefix = team_id or "local"
    return (
        ParsedMessage(
            message_id=f"{prefix}:{channel_id}:{ts}",
            channel_id=channel_id,
            user_id=user_id,
            text=text,
            ts=ts,
            raw=event,
        ),
        "accepted",
    )


class SlackPriorityService:
    def __init__(self, settings: Settings, store: Store | None = None):
        if not settings.slack_user_token:
            raise RuntimeError("Missing Slack user token. Run setup or set SLACK_USER_TOKEN.")
        if not settings.slack_app_token:
            raise RuntimeError("Missing Slack app token. Run setup or set SLACK_APP_TOKEN.")

        self.settings = settings
        self.web = WebClient(token=settings.slack_user_token)
        self.store = store or Store()
        self.scorer = UrgencyScorer(thresholds=settings.thresholds)
        self.team_id, self.self_user_id = self._auth_identity()
        self._stop_event = threading.Event()
        self._socket: SocketModeClient | None = None
        self._on_status: Callable[[str], None] | None = None
        self._membership_cache: dict[str, bool] = {}
        self._channel_name_cache: dict[str, str] = {}

    def backfill(
        self,
        *,
        days: int = 14,
        max_conversations: int | None = None,
        conversation_types: str = CONVERSATION_TYPES,
        channel_ids: list[str] | None = None,
        on_status: Callable[[str], None] | None = None,
        verbose: bool = False,
    ) -> int:
        oldest = str(time.time() - days * 24 * 60 * 60)
        imported = 0
        messages_seen = 0
        skipped: Counter[str] = Counter()
        conversations = (
            [{"id": channel_id, "name": channel_id} for channel_id in channel_ids]
            if channel_ids
            else self._list_conversations(types=conversation_types)
        )
        conversations = [
            conversation for conversation in conversations
            if self._conversation_is_joined(conversation)
        ]
        if max_conversations is not None:
            conversations = conversations[:max_conversations]

        if on_status:
            on_status(f"Backfilling {len(conversations)} conversations from the last {days} days...")

        for index, conversation in enumerate(conversations, start=1):
            channel_id = conversation["id"]
            channel_name = conversation.get("name") or conversation.get("user") or channel_id
            cursor = None
            imported_in_channel = 0
            seen_in_channel = 0
            while True:
                try:
                    response = self.web.conversations_history(
                        channel=channel_id,
                        oldest=oldest,
                        limit=200,
                        cursor=cursor,
                    )
                except SlackApiError as error:
                    if error.response.status_code == 429:
                        retry_after = int(error.response.headers.get("Retry-After", "60"))
                        time.sleep(retry_after)
                        continue
                    if on_status:
                        on_status(f"Skipped {channel_name}: {error.response.get('error')}")
                    break

                messages = response.get("messages", [])
                messages_seen += len(messages)
                seen_in_channel += len(messages)
                for event in messages:
                    parsed, reason = classify_message_event(
                        event,
                        team_id=self.team_id,
                        self_user_id=self.self_user_id,
                        fallback_channel_id=channel_id,
                    )
                    if not parsed:
                        skipped[reason] += 1
                        continue
                    self._store_and_score(parsed, channel_name=channel_name, fetch_permalink=False)
                    imported += 1
                    imported_in_channel += 1

                cursor = response.get("response_metadata", {}).get("next_cursor")
                if not cursor:
                    break

            if on_status and (verbose or imported_in_channel > 0):
                on_status(
                    f"{channel_name}: saw {seen_in_channel}, imported {imported_in_channel} "
                    f"(total imported {imported})"
                )
            elif on_status and index % 25 == 0:
                on_status(
                    f"Scanned {index}/{len(conversations)} conversations; "
                    f"saw {messages_seen}, imported {imported}"
                )

        if on_status:
            skipped_text = ", ".join(
                f"{reason}={count}" for reason, count in skipped.most_common()
            )
            on_status(
                f"Backfill complete: scanned {len(conversations)} conversations, "
                f"saw {messages_seen} Slack messages, imported {imported} user messages."
            )
            if skipped_text:
                on_status(f"Skipped messages: {skipped_text}")

        return imported

    def listen_forever(self, *, on_status: Callable[[str], None] | None = None) -> None:
        self._on_status = on_status
        self._stop_event.clear()
        if on_status:
            on_status("Slack listener starting...")
        self._socket = SocketModeClient(
            app_token=self.settings.slack_app_token,
            web_client=self.web,
        )
        self._socket.socket_mode_request_listeners.append(self._handle_socket_request)
        self._socket.connect()
        if on_status:
            on_status("Slack listener connected")

        while not self._stop_event.wait(1):
            pass

        if hasattr(self._socket, "close"):
            self._socket.close()
        elif hasattr(self._socket, "disconnect"):
            self._socket.disconnect()

    def stop(self) -> None:
        self._stop_event.set()

    def _handle_socket_request(self, client: SocketModeClient, request: SocketModeRequest) -> None:
        client.send_socket_mode_response(SocketModeResponse(envelope_id=request.envelope_id))
        if request.type != "events_api":
            return

        payload = request.payload or {}
        event = payload.get("event", {})
        parsed, reason = classify_message_event(
            event,
            team_id=payload.get("team_id"),
            self_user_id=self.self_user_id,
        )
        if self._on_status:
            self._on_status(
                "event "
                f"type={event.get('type')} "
                f"subtype={event.get('subtype')} "
                f"channel={event.get('channel')} "
                f"reason={reason} "
                f"{_event_log_details(event)}"
            )
        if reason == "self_message":
            self._schedule_thread_scan(event, team_id=payload.get("team_id"), reason=reason)
        elif event.get("subtype") == "message_replied":
            self._schedule_thread_scan(event, team_id=payload.get("team_id"), reason="message_replied")

        if parsed:
            if not self._channel_is_joined(parsed.channel_id):
                if self._on_status:
                    self._on_status(f"skipped channel={parsed.channel_id} reason=not_joined")
                return
            try:
                self._store_and_score(
                    parsed,
                    channel_name=self._channel_display_name(parsed.channel_id),
                    fetch_permalink=False,
                )
            except Exception as error:
                if self._on_status:
                    self._on_status(f"store_failed channel={parsed.channel_id} error={error}")

    def _store_and_score(
        self,
        parsed: ParsedMessage,
        *,
        channel_name: str | None = None,
        fetch_permalink: bool = True,
    ) -> None:
        permalink = self._permalink(parsed.channel_id, parsed.ts) if fetch_permalink else None
        is_new_message = not self.store.message_exists(parsed.message_id)
        self.store.upsert_message(
            message_id=parsed.message_id,
            text=parsed.text,
            channel_id=parsed.channel_id,
            channel_name=channel_name,
            user_id=parsed.user_id,
            ts=parsed.ts,
            permalink=permalink,
            raw=parsed.raw,
        )

        score = self.scorer.score(parsed.text)
        level = score_to_level(score, self.settings.thresholds)
        self.store.upsert_prediction(
            message_id=parsed.message_id,
            score=score,
            level=level,
            model_version=MODEL_VERSION,
        )

        if is_new_message and should_notify(level, self.settings.notify_min_level):
            if not permalink:
                permalink = self._permalink(parsed.channel_id, parsed.ts)
                if permalink:
                    self.store.upsert_message(
                        message_id=parsed.message_id,
                        text=parsed.text,
                        channel_id=parsed.channel_id,
                        channel_name=channel_name,
                        user_id=parsed.user_id,
                        ts=parsed.ts,
                        permalink=permalink,
                        raw=parsed.raw,
                    )
            notify_url = slack_channel_url(team_id=getattr(self, "team_id", None), channel_id=parsed.channel_id) or permalink
            notify_result = send_notification(
                title=f"Slack {level}: {score:.2f}",
                message=_shorten(parsed.text),
                url=notify_url,
            )
            on_status = getattr(self, "_on_status", None)
            if on_status:
                status = "sent" if notify_result.ok else "failed"
                detail = f" stderr={notify_result.stderr}" if notify_result.stderr else ""
                on_status(
                    f"notify_{status} method={notify_result.method} "
                    f"code={notify_result.returncode} url={notify_url}{detail}"
                )
        on_status = getattr(self, "_on_status", None)
        if on_status:
            on_status(
                f"stored new={is_new_message} score={score:.3f} "
                f"level={level} text={_shorten(parsed.text, 80)}"
            )

    def _schedule_thread_scan(self, event: dict, *, team_id: str | None, reason: str) -> None:
        channel_id = str(event.get("channel") or "")
        thread_ts = _thread_ts(event)
        if not channel_id or not thread_ts:
            return
        if not self._channel_is_joined(channel_id):
            if self._on_status:
                self._on_status(f"skipped thread_scan channel={channel_id} reason=not_joined")
            return

        if self._on_status:
            self._on_status(f"thread_scan scheduled channel={channel_id} ts={thread_ts} reason={reason}")

        def scan_later() -> None:
            for delay in (2.0, 5.0, 10.0, 20.0, 45.0):
                if self._stop_event.wait(delay):
                    return
                stored = self._capture_thread_replies(
                    channel_id=channel_id,
                    thread_ts=thread_ts,
                    team_id=team_id,
                )
                if stored > 0:
                    return

        threading.Thread(target=scan_later, name="slack-thread-scan", daemon=True).start()

    def _capture_thread_replies(self, *, channel_id: str, thread_ts: str, team_id: str | None) -> int:
        try:
            response = self.web.conversations_replies(
                channel=channel_id,
                ts=thread_ts,
                limit=100,
            )
        except SlackApiError as error:
            if self._on_status:
                self._on_status(
                    f"thread_scan_failed channel={channel_id} ts={thread_ts} "
                    f"error={error.response.get('error')}"
                )
            return 0

        messages = response.get("messages", [])
        stored = 0
        skipped: Counter[str] = Counter()
        for message in messages:
            parsed, reason = classify_message_event(
                message,
                team_id=team_id or self.team_id,
                self_user_id=self.self_user_id,
                fallback_channel_id=channel_id,
            )
            if not parsed:
                skipped[reason] += 1
                if self._on_status:
                    self._on_status(f"thread_skip reason={reason} {_event_log_details(message)}")
                continue
            if parsed.ts == thread_ts and parsed.user_id == self.self_user_id:
                skipped["thread_parent_self"] += 1
                if self._on_status:
                    self._on_status(f"thread_skip reason=thread_parent_self {_event_log_details(message)}")
                continue
            try:
                self._store_and_score(
                    parsed,
                    channel_name=self._channel_display_name(channel_id),
                    fetch_permalink=False,
                )
                stored += 1
            except Exception as error:
                if self._on_status:
                    self._on_status(f"store_failed channel={channel_id} error={error}")

        if self._on_status:
            skipped_text = ", ".join(f"{key}={value}" for key, value in skipped.most_common())
            suffix = f" skipped={skipped_text}" if skipped_text else ""
            self._on_status(
                f"thread_scan channel={channel_id} ts={thread_ts} "
                f"saw={len(messages)} stored={stored}{suffix}"
            )
        return stored

    def _list_conversations(self, *, types: str = CONVERSATION_TYPES) -> list[dict]:
        conversations = []
        cursor = None
        while True:
            response = self.web.conversations_list(
                types=types,
                exclude_archived=True,
                limit=200,
                cursor=cursor,
            )
            conversations.extend(response.get("channels", []))
            cursor = response.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                return conversations

    def list_conversations(self, *, types: str = CONVERSATION_TYPES, limit: int = 50) -> list[dict]:
        return [
            conversation for conversation in self._list_conversations(types=types)
            if self._conversation_is_joined(conversation)
        ][:limit]

    def debug_channel_history(
        self,
        *,
        channel_id: str,
        days: int = 14,
        latest_ts: str | None = None,
    ) -> dict:
        oldest = str(time.time() - days * 24 * 60 * 60)
        latest = latest_ts or str(time.time())
        info = None
        history = None
        try:
            info = self.web.conversations_info(channel=channel_id).data
        except SlackApiError as error:
            info = {
                "ok": False,
                "error": error.response.get("error"),
                "needed": error.response.get("needed"),
                "provided": error.response.get("provided"),
            }

        try:
            response = self.web.conversations_history(
                channel=channel_id,
                oldest=oldest,
                latest=latest,
                inclusive=True,
                limit=5,
            )
            messages = response.get("messages", [])
            history = {
                "ok": response.get("ok"),
                "message_count": len(messages),
                "has_more": response.get("has_more"),
                "next_cursor": bool(response.get("response_metadata", {}).get("next_cursor")),
                "messages": [
                    {
                        "type": message.get("type"),
                        "subtype": message.get("subtype"),
                        "has_text": bool(message.get("text")),
                        "has_user": bool(message.get("user")),
                        "has_channel": bool(message.get("channel")),
                        "ts": message.get("ts"),
                        "text_preview": str(message.get("text") or "")[:120],
                    }
                    for message in messages
                ],
            }
        except SlackApiError as error:
            history = {
                "ok": False,
                "error": error.response.get("error"),
                "needed": error.response.get("needed"),
                "provided": error.response.get("provided"),
                "status_code": error.response.status_code,
            }

        return {
            "channel_id": channel_id,
            "days": days,
            "oldest": oldest,
            "latest": latest,
            "info": info,
            "history": history,
        }

    def _permalink(self, channel_id: str, ts: str) -> str | None:
        try:
            response = self.web.chat_getPermalink(channel=channel_id, message_ts=ts)
        except SlackApiError:
            return None
        return response.get("permalink")

    def close(self) -> None:
        self.store.close()

    def _conversation_is_joined(self, conversation: dict) -> bool:
        channel_id = str(conversation.get("id") or "")
        if channel_id:
            name = conversation.get("name") or conversation.get("user")
            if name:
                self._channel_name_cache[channel_id] = str(name)
        if conversation.get("is_im") or conversation.get("is_mpim"):
            return True
        if conversation.get("is_member") is True:
            return True
        return bool(channel_id and self._channel_is_joined(channel_id))

    def _channel_is_joined(self, channel_id: str) -> bool:
        if channel_id in self._membership_cache:
            return self._membership_cache[channel_id]
        if channel_id.startswith("D"):
            self._membership_cache[channel_id] = True
            return True

        try:
            response = self.web.conversations_info(channel=channel_id)
        except SlackApiError:
            self._membership_cache[channel_id] = False
            return False

        channel = response.get("channel", {})
        name = channel.get("name") or channel.get("user")
        if name:
            self._channel_name_cache[channel_id] = str(name)
        is_joined = bool(
            channel.get("is_im")
            or channel.get("is_mpim")
            or channel.get("is_member")
        )
        self._membership_cache[channel_id] = is_joined
        return is_joined

    def _refresh_joined_channels(self) -> None:
        try:
            conversations = self._list_conversations(types=CONVERSATION_TYPES)
        except SlackApiError as error:
            if self._on_status:
                self._on_status(f"membership_refresh_failed error={error.response.get('error')}")
            return

        joined = 0
        for conversation in conversations:
            channel_id = str(conversation.get("id") or "")
            if not channel_id:
                continue
            name = conversation.get("name") or conversation.get("user")
            if name:
                self._channel_name_cache[channel_id] = str(name)
            is_joined = bool(
                conversation.get("is_im")
                or conversation.get("is_mpim")
                or conversation.get("is_member")
            )
            self._membership_cache[channel_id] = is_joined
            if is_joined:
                joined += 1

        if self._on_status:
            self._on_status(f"membership loaded: {joined}/{len(conversations)} joined conversations")

    def _auth_identity(self) -> tuple[str | None, str | None]:
        try:
            response = self.web.auth_test()
        except SlackApiError:
            return None, None
        return response.get("team_id"), response.get("user_id")

    def _channel_display_name(self, channel_id: str) -> str | None:
        return self._channel_name_cache.get(channel_id)


def _shorten(text: str, limit: int = 180) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "..."


def _thread_ts(event: dict) -> str:
    direct = event.get("thread_ts") or event.get("ts")
    if direct:
        return str(direct)

    nested = event.get("message")
    if isinstance(nested, dict):
        nested_ts = nested.get("thread_ts") or nested.get("ts")
        if nested_ts:
            return str(nested_ts)
    return ""


def _event_log_details(event: dict) -> str:
    parts = [
        f"user={event.get('user')}",
        f"bot={event.get('bot_id')}",
        f"app={event.get('app_id')}",
        f"ts={event.get('ts')}",
    ]
    thread_ts = event.get("thread_ts")
    if thread_ts:
        parts.append(f"thread_ts={thread_ts}")
    parts.append(f"text={_shorten(str(event.get('text') or ''), 120)!r}")
    return " ".join(parts)


def pretty_event(event: dict) -> str:
    return json.dumps(event, indent=2, sort_keys=True)
