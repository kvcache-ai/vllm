# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

_MOONCAKE_IMPORT_ERROR: ImportError | None
try:
    from mooncake.engine import TransferEngine as _TransferEngine  # noqa: F401
except ImportError as e:
    _MOONCAKE_IMPORT_ERROR = e
else:
    _MOONCAKE_IMPORT_ERROR = None


def ensure_mooncake_available() -> None:
    if _MOONCAKE_IMPORT_ERROR is not None:
        raise ImportError(
            "Install mooncake-transfer-engine (see "
            "https://github.com/kvcache-ai/Mooncake ) to use ECMooncakeConnector."
        ) from _MOONCAKE_IMPORT_ERROR
