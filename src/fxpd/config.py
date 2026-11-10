"""Central configuration: instruments, triangle identities, and on-disk paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# parents[2] == project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
QC_DIR = DATA_DIR / "qc"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

@dataclass(frozen=True)
class Triangle:
    """A no-arbitrage triangle.  Identity in logs: cross = leg1 + leg2."""

    name: str
    cross: str
    leg1: str
    leg2: str

    @property
    def legs(self) -> tuple[str, str, str]:
        """All three instruments, cross first."""
        return (self.cross, self.leg1, self.leg2)

TRIANGLES: dict[str, Triangle] = {
    "majors": Triangle("majors", cross="EURJPY", leg1="EURUSD", leg2="USDJPY"),
    "gbp": Triangle("gbp", cross="EURUSD", leg1="EURGBP", leg2="GBPUSD"),
}

def ensure_dirs() -> None:
    """Create the data/output directory tree if missing."""
    for d in (RAW_DIR, PROCESSED_DIR, QC_DIR, OUTPUTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
