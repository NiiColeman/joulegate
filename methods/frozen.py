"""Never update: zero adaptation energy, the accuracy floor."""
from methods.base import Policy


class Frozen(Policy):
    adapts = False

    def should_update(self, t, ctx):
        return False
