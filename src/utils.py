import pyRAPL
import json
import joblib
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import subprocess
import threading
import time
import logging

import pyRAPL

class EnergyTracker:
    def __enter__(self):
        pyRAPL.setup()
        self.meter = pyRAPL.Measurement('energy')
        self.meter.begin()
        return self
    def __exit__(self, *args):
        self.meter.end()
        self.energy_joules = self.meter.result.energy
    def get_energy(self):
        return self.energy_joules