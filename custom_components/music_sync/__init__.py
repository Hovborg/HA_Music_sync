"""
Music Sync - Home Assistant Integration

Synchronizes audio playback across different speaker platforms
(Google Cast, Sonos, Alexa) using microphone-based latency detection.
"""
from __future__ import annotations

import logging
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers import device_registry as dr

from .const import (
    DOMAIN,
    PLATFORMS,
    SERVICE_CALIBRATE,
    SERVICE_CALIBRATE_SPEAKER,
    SERVICE_SET_OFFSET,
    SERVICE_CLEAR_OFFSETS,
    CONF_SPEAKERS,
    ATTR_OFFSET_MS,
)
from .coordinator import MusicSyncCoordinator

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Music Sync from a config entry."""
    _LOGGER.info("Setting up Music Sync integration")

    # Create coordinator
    coordinator = MusicSyncCoordinator(hass, entry)

    # Store coordinator in hass.data
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = coordinator

    # Initialize coordinator
    await coordinator.async_initialize()

    # Set up platforms
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    # Register services
    await async_setup_services(hass, coordinator)

    # Register update listener for options
    entry.async_on_unload(entry.add_update_listener(async_update_options))

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    _LOGGER.info("Unloading Music Sync integration")

    # Unload platforms
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

    if unload_ok:
        # Cleanup coordinator
        coordinator: MusicSyncCoordinator = hass.data[DOMAIN].pop(entry.entry_id)
        await coordinator.async_shutdown()

    return unload_ok


async def async_update_options(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle options update."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_setup_services(hass: HomeAssistant, coordinator: MusicSyncCoordinator) -> None:
    """Set up services for Music Sync."""

    async def handle_calibrate(call: ServiceCall) -> None:
        """Handle the calibrate service call."""
        _LOGGER.info("Starting full calibration")
        await coordinator.async_calibrate_all()

    async def handle_calibrate_speaker(call: ServiceCall) -> None:
        """Handle the calibrate_speaker service call."""
        entity_id = call.data.get("entity_id")
        _LOGGER.info("Calibrating speaker: %s", entity_id)
        await coordinator.async_calibrate_speaker(entity_id)

    async def handle_set_offset(call: ServiceCall) -> None:
        """Handle the set_offset service call."""
        entity_id = call.data.get("entity_id")
        offset_ms = call.data.get(ATTR_OFFSET_MS)
        _LOGGER.info("Setting offset for %s: %sms", entity_id, offset_ms)
        await coordinator.async_set_offset(entity_id, offset_ms)

    async def handle_clear_offsets(call: ServiceCall) -> None:
        """Handle the clear_offsets service call."""
        _LOGGER.info("Clearing all offsets")
        await coordinator.async_clear_offsets()

    # Register services
    hass.services.async_register(DOMAIN, SERVICE_CALIBRATE, handle_calibrate)
    hass.services.async_register(DOMAIN, SERVICE_CALIBRATE_SPEAKER, handle_calibrate_speaker)
    hass.services.async_register(DOMAIN, SERVICE_SET_OFFSET, handle_set_offset)
    hass.services.async_register(DOMAIN, SERVICE_CLEAR_OFFSETS, handle_clear_offsets)
