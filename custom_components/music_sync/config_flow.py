"""Config flow for Music Sync integration."""
from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol

from homeassistant.config_entries import ConfigEntry, ConfigFlow, OptionsFlow
from homeassistant.core import callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector
from homeassistant.components.media_player import DOMAIN as MEDIA_PLAYER_DOMAIN

from .const import (
    DOMAIN,
    CONF_SPEAKERS,
    CONF_MICROPHONE_DEVICE,
    CONF_SAMPLE_RATE,
    CONF_CALIBRATION_INTERVAL,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_CALIBRATION_INTERVAL,
)

_LOGGER = logging.getLogger(__name__)


class MusicSyncConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Music Sync."""

    VERSION = 1

    def __init__(self) -> None:
        """Initialize the config flow."""
        self._speakers: list[str] = []
        self._microphone_device: str | None = None

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step - select speakers."""
        errors: dict[str, str] = {}

        if user_input is not None:
            self._speakers = user_input.get(CONF_SPEAKERS, [])

            if len(self._speakers) < 2:
                errors["base"] = "min_speakers"
            else:
                # Proceed to microphone setup
                return await self.async_step_microphone()

        # Get available media players
        media_players = self.hass.states.async_entity_ids(MEDIA_PLAYER_DOMAIN)

        if not media_players:
            return self.async_abort(reason="no_media_players")

        schema = vol.Schema(
            {
                vol.Required(CONF_SPEAKERS): selector.EntitySelector(
                    selector.EntitySelectorConfig(
                        domain=MEDIA_PLAYER_DOMAIN,
                        multiple=True,
                    )
                ),
            }
        )

        return self.async_show_form(
            step_id="user",
            data_schema=schema,
            errors=errors,
        )

    async def async_step_microphone(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle microphone configuration step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            self._microphone_device = user_input.get(CONF_MICROPHONE_DEVICE, "default")

            # Create the config entry
            return self.async_create_entry(
                title="Music Sync",
                data={
                    CONF_SPEAKERS: self._speakers,
                    CONF_MICROPHONE_DEVICE: self._microphone_device,
                },
                options={
                    CONF_SAMPLE_RATE: DEFAULT_SAMPLE_RATE,
                    CONF_CALIBRATION_INTERVAL: DEFAULT_CALIBRATION_INTERVAL,
                },
            )

        # Simple text input for microphone device
        schema = vol.Schema(
            {
                vol.Optional(CONF_MICROPHONE_DEVICE, default="default"): str,
            }
        )

        return self.async_show_form(
            step_id="microphone",
            data_schema=schema,
            errors=errors,
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlow:
        """Get the options flow for this handler."""
        return MusicSyncOptionsFlow(config_entry)


class MusicSyncOptionsFlow(OptionsFlow):
    """Handle options flow for Music Sync."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage the options."""
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        current_options = self.config_entry.options

        schema = vol.Schema(
            {
                vol.Optional(
                    CONF_SAMPLE_RATE,
                    default=current_options.get(CONF_SAMPLE_RATE, DEFAULT_SAMPLE_RATE),
                ): vol.All(vol.Coerce(int), vol.Range(min=8000, max=96000)),
                vol.Optional(
                    CONF_CALIBRATION_INTERVAL,
                    default=current_options.get(
                        CONF_CALIBRATION_INTERVAL, DEFAULT_CALIBRATION_INTERVAL
                    ),
                ): vol.All(vol.Coerce(int), vol.Range(min=300, max=86400)),
            }
        )

        return self.async_show_form(step_id="init", data_schema=schema)
