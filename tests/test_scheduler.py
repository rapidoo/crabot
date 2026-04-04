"""Tests for core/scheduler.py — ScheduledTask and AgentScheduler."""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.core.scheduler import AgentScheduler, Event, ScheduledTask


def test_scheduled_task_is_due():
    task = ScheduledTask(
        name="test",
        prompt="hello",
        schedule_type="once",
        schedule_value="",
        next_run=datetime.now() - timedelta(seconds=1),
    )
    assert task.is_due()


def test_scheduled_task_not_due():
    task = ScheduledTask(
        name="test",
        prompt="hello",
        schedule_type="once",
        schedule_value="",
        next_run=datetime.now() + timedelta(hours=1),
    )
    assert not task.is_due()


def test_scheduled_task_advance_once():
    task = ScheduledTask(
        name="test",
        prompt="hello",
        schedule_type="once",
        schedule_value="",
    )
    task.advance()
    assert not task.enabled


def test_scheduled_task_advance_interval():
    now = datetime.now()
    task = ScheduledTask(
        name="test",
        prompt="hello",
        schedule_type="interval",
        schedule_value="60",
        next_run=now,
    )
    task.advance()
    assert task.next_run == now + timedelta(seconds=60)
    assert task.enabled


def test_scheduler_add_task():
    agent = MagicMock()
    scheduler = AgentScheduler(agent)
    task = ScheduledTask(name="t", prompt="p", schedule_type="once", schedule_value="")
    scheduler.add_task(task)
    assert scheduler.task_count == 1


def test_scheduler_schedule_self():
    agent = MagicMock()
    scheduler = AgentScheduler(agent)
    scheduler.schedule_self("do something", delay_seconds=300)
    assert scheduler.task_count == 1


@pytest.mark.asyncio
async def test_scheduler_push_event():
    agent = MagicMock()
    scheduler = AgentScheduler(agent)
    await scheduler.push_event(Event(prompt="test", source="user"))
    assert not scheduler._queue.empty()
