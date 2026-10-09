from typing import List
from enum import Enum

from modules.processing import StableDiffusionProcessing

from internal_controlnet.external_code import ControlNetUnit
from scripts.logging import logger


class Infotext:
    """ControlNet's generation-infotext contract: one "ControlNet <i>" entry per enabled unit, and its parse for
    /sdapi/v1/png-info (on_infotext_pasted)."""

    @staticmethod
    def unit_prefix(unit_index: int) -> str:
        return f"ControlNet {unit_index}"

    @staticmethod
    def write_infotext(units: List[ControlNetUnit], p: StableDiffusionProcessing):
        """Write infotext to `p`."""
        p.extra_generation_params.update(
            {
                Infotext.unit_prefix(i): unit.serialize()
                for i, unit in enumerate(units)
                if unit.enabled
            }
        )

    @staticmethod
    def on_infotext_pasted(infotext: str, results: dict) -> None:
        """Parse ControlNet infotext string and write result to `results` dict."""
        updates = {}
        for k, v in results.items():
            if not k.startswith("ControlNet"):
                continue

            assert isinstance(v, str), f"Expect string but got {v}."
            try:
                for field, value in vars(ControlNetUnit.parse(v)).items():
                    if field not in ControlNetUnit.infotext_fields():
                        continue
                    if value is None:
                        logger.debug(
                            f"InfoText: Skipping {field} because value is None."
                        )
                        continue

                    component_locator = f"{k} {field}"
                    if isinstance(value, Enum):
                        value = value.value

                    updates[component_locator] = value
                    logger.debug(f"InfoText: Setting {component_locator} = {value}")
            except Exception as e:
                logger.warning(
                    f"Failed to parse infotext, legacy format infotext is no longer supported:\n{v}\n{e}"
                )

        results.update(updates)
