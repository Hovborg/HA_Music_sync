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
            description_placeholders={
                "speaker_count": str(len(media_players)),
            },
        )

    async def async_step_microphone(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle microphone configuration step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            self._microphone_device = user_input.get(CONF_MICROPHONE_DEVICE)

            # Validate microphone (in a real implementation)
            # For now, accept any input

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

        # Get available audio input devices
        microphone_options = await self._get_microphone_options()

        schema = vol.Schema(
            {
                vol.Required(CONF_MICROPHONE_DEVICE): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=microphone_options,
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                ),
            }
        )

        return self.async_show_form(
            step_id="microphone",
            data_schema=schema,
            errors=errors,
            description_placeholders={
                "speaker_list": ", ".join(self._speakers),
            },
        )

    async def _get_microphone_options(self) -> list[selector.SelectOptionDict]:
        """Get available microphone devices."""
        options: list[selector.SelectOptionDict] = []

        try:
            import sounddevice as sd

            devices = sd.query_devices()
            for idx, device in enumerate(devices):
                if device.get("max_input_channels", 0) > 0:
                    options.append(
                        selector.SelectOptionDict(
                            value=str(idx),
                            label=f"{device['name']} ({device['max_input_channels']} ch)",
                        )
                    )
        except Exception as err:
            _LOGGER.warning("Could not enumerate audio devices: %s", err)

        # Add default option
        if not options:
            options.append(
                selector.SelectOptionDict(
                    value="default",
                    label="System Default Microphone",
                )
            )

        return options

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
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=8000,
                        max=96000,
                        step=1000,
                        mode=selector.NumberSelectorMode.BOX,
                        unit_of_measurement="Hz",
                    )
                ),
                vol.Optional(
                    CONF_CALIBRATION_INTERVAL,
                    default=current_options.get(
                        CONF_CALIBRATION_INTERVAL, DEFAULT_CALIBRATION_INTERVAL
                    ),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=300,
                        max=86400,
                        step=300,
                        mode=selector.NumberSelectorMode.BOX,
                        unit_of_measurement="seconds",
                    )
                ),
            }
        )

        return self.async_show_form(step_id="init", data_schema=schema)
