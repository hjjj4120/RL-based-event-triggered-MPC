import math
import numpy as np

try:
    import casadi as ca
except ImportError:
    ca = None


def _lib(x):
    if ca is not None and "casadi" in type(x).__module__:
        return ca
    return np


class PathSpec:
    def __init__(self, name, params, ref_func):
        self.name = name
        self.params = params
        self.ref_func = ref_func

    def ref(self, x):
        return self.ref_func(x)


def dang_single(amplitude=4.0, wavelength=50.0):
    def ref(x):
        lib = _lib(x)
        return amplitude * lib.sin(2 * lib.pi / wavelength * x)
    return PathSpec("dang_single", {"amplitude": amplitude, "wavelength": wavelength}, ref)


def sine(amplitude=2.5, wavelength=40.0):
    def ref(x):
        lib = _lib(x)
        return amplitude * lib.sin(2 * lib.pi / wavelength * x)
    return PathSpec("sine", {"amplitude": amplitude, "wavelength": wavelength}, ref)


def double_lane_change(amplitude=3.5, sharpness=0.25, x1=15.0, x2=35.0):
    def ref(x):
        lib = _lib(x)
        return amplitude * (lib.tanh(sharpness * (x - x1)) - lib.tanh(sharpness * (x - x2)))
    return PathSpec(
        "double_lane_change",
        {"amplitude": amplitude, "sharpness": sharpness, "x1": x1, "x2": x2},
        ref,
    )


def clothoid_turn(curvature=0.002, scale=1.0):
    def ref(x):
        lib = _lib(x)
        return scale * curvature * x * x
    return PathSpec("clothoid_turn", {"curvature": curvature, "scale": scale}, ref)


def random_spline(rng, amplitude=2.0, base_freq=0.08, harmonics=3):
    phases = rng.uniform(0, 2 * math.pi, size=harmonics)
    weights = rng.uniform(0.5, 1.0, size=harmonics)

    def ref(x):
        lib = _lib(x)
        y = 0
        for idx in range(harmonics):
            y = y + weights[idx] * lib.sin((idx + 1) * base_freq * x + phases[idx])
        return amplitude * y

    return PathSpec(
        "random_spline",
        {"amplitude": amplitude, "base_freq": base_freq, "harmonics": harmonics},
        ref,
    )


class PathSampler:
    def __init__(self, seed=0):
        self.rng = np.random.RandomState(seed)

    def sample(self, path_name, domain_rand=False):
        if path_name == "dang_single":
            amp = 4.0
            wave = 50.0
            if domain_rand:
                amp *= self.rng.uniform(0.9, 1.1)
                wave *= self.rng.uniform(0.9, 1.1)
            return dang_single(amp, wave)
        if path_name == "sine":
            amp = 2.5
            wave = 40.0
            if domain_rand:
                amp *= self.rng.uniform(0.8, 1.2)
                wave *= self.rng.uniform(0.85, 1.15)
            return sine(amp, wave)
        if path_name == "double_lane_change":
            amp = 3.5
            sharp = 0.25
            x1 = 15.0
            x2 = 35.0
            if domain_rand:
                amp *= self.rng.uniform(0.9, 1.1)
                sharp *= self.rng.uniform(0.8, 1.2)
                shift = self.rng.uniform(-2.0, 2.0)
                x1 += shift
                x2 += shift
            return double_lane_change(amp, sharp, x1, x2)
        if path_name == "clothoid_turn":
            curvature = 0.002
            scale = 1.0
            if domain_rand:
                curvature *= self.rng.uniform(0.8, 1.2)
                scale *= self.rng.uniform(0.9, 1.1)
            return clothoid_turn(curvature, scale)
        if path_name == "random_spline":
            amp = 2.0
            base_freq = 0.08
            harmonics = 3
            if domain_rand:
                amp *= self.rng.uniform(0.8, 1.2)
                base_freq *= self.rng.uniform(0.8, 1.2)
            return random_spline(self.rng, amp, base_freq, harmonics)
        raise ValueError(f"Unknown path name: {path_name}")


def curriculum_path_pool(progress):
    if progress < 0.2:
        return ["dang_single"]
    if progress < 0.6:
        return ["dang_single", "sine", "double_lane_change"]
    return ["dang_single", "sine", "double_lane_change", "clothoid_turn", "random_spline"]


def all_paths():
    return ["dang_single", "sine", "double_lane_change", "clothoid_turn", "random_spline"]
