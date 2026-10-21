"""Labels the API will accept.

Imports from shared.labels rather than re-declaring the list. The v1 version of
this file kept its own copy, which is exactly how a vocabulary drifts out of
sync between the server, the agent, and the trainer.
"""
from shared.labels import ALL_CODES

VALID_LABELS = list(ALL_CODES)
