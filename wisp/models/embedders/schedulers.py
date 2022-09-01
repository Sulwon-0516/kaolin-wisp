import abc
import torch
import math
import copy
import collections
from typing import Any, Iterable, List, Tuple, Union



def from_tuple(x):
  schedule_type, *args = x
  return SCHEDULE_MAP[schedule_type](*args)


def from_dict(d):
  d = copy.copy(dict(d))
  schedule_type = d.pop('type')
  return SCHEDULE_MAP[schedule_type](**d)


def from_config(schedule):
  """Creates a schedule from a configuration."""
  if schedule is None:
    return NoneSchedule()
  if isinstance(schedule, Schedule):
    return schedule
  if isinstance(schedule, Tuple) or isinstance(schedule, List):
    return from_tuple(schedule)
  if isinstance(schedule, collections.Mapping):
    return from_dict(schedule)

  raise ValueError(f'Unknown type {type(schedule)}.')

class Schedule(abc.ABC):
  """An interface for generic schedules.."""

  @abc.abstractmethod
  def get(self, step):
    """Get the value for the given step."""
    raise NotImplementedError

  def __call__(self, step):
    return self.get(step)


class NoneSchedule(Schedule):
  """Always returns None. Useful for disable schedules."""

  def get(self, step):
    return None


class ConstantSchedule(Schedule):
  """Linearly scaled scheduler."""

  def __init__(self, value):
    # you should put it as raw float (not tensor)
    super().__init__()
    self.value = value

  def get(self, step):
    """Get the value for the given step."""
    if self.value is None:
      return None
    if isinstance(step, torch.Tensor):
        step = step.item()

    return self.value


class LinearSchedule(Schedule):
  """Linearly scaled scheduler."""

  def __init__(self, initial_value, final_value, num_steps):
    # you should put it as raw float / int
    super().__init__()
    self.initial_value = initial_value
    self.final_value = final_value
    self.num_steps = num_steps

  def get(self, step):
    """Get the value for the given step."""
    if isinstance(step, torch.Tensor):
        step = step.item()
    
    if self.num_steps == 0:
      return self.final_value
    alpha = max(step / self.num_steps, 1.0)
    return (1.0 - alpha) * self.initial_value + alpha * self.final_value


class ExponentialSchedule(Schedule):
  """Exponentially decaying scheduler."""

  def __init__(self, initial_value, final_value, num_steps, eps=1e-10):
    super().__init__()
    if initial_value <= final_value:
      raise ValueError('Final value must be less than initial value.')

    self.initial_value = initial_value
    self.final_value = final_value
    self.num_steps = num_steps
    if self.num_steps == 1:
        print("It doesn't allow single step")
        assert()
    self.eps = eps

  def get(self, step):
    """Get the value for the given step."""
    if isinstance(step, torch.Tensor):
        step = step.item()

    if step >= self.num_steps:
      return self.final_value

    final_value = max(self.final_value, self.eps)
    base = final_value / self.initial_value
    exponent = step / (self.num_steps - 1)

    return self.initial_value * base**exponent


class CosineEasingSchedule(Schedule):
  """Schedule that eases slowly using a cosine."""

  def __init__(self, initial_value, final_value, num_steps):
    super().__init__()
    self.initial_value = initial_value
    self.final_value = final_value
    self.num_steps = num_steps

  def get(self, step):
    """Get the value for the given step."""
    if isinstance(step, torch.Tensor):
        step = step.item()

    alpha = min(step / self.num_steps, 1.0)
    scale = self.final_value - self.initial_value
    x = min(max(alpha, 0.0), 1.0)
    return (self.initial_value + scale * 0.5 * (1 + math.cos(math.pi * x + math.pi)))


class StepSchedule(Schedule):
  """Schedule that eases slowsly using a cosine."""

  def __init__(self,
               initial_value,
               decay_interval,
               decay_factor,
               max_decays,
               final_value=None):
    super().__init__()
    self.initial_value = initial_value
    self.decay_factor = decay_factor
    self.decay_interval = decay_interval
    self.max_decays = max_decays
    if final_value is None:
      final_value = self.initial_value * self.decay_factor**self.max_decays
    self.final_value = final_value

  def get(self, step):
    """Get the value for the given step."""
    if isinstance(step, torch.Tensor):
        step = step.item()

    
    phase = step // self.decay_interval
    if phase >= self.max_decays:
      return self.final_value
    else:
      return self.initial_value * self.decay_factor**phase


class PiecewiseSchedule(Schedule):
  """A piecewise combination of multiple schedules."""

  def __init__(
      self, schedules: Iterable[Tuple[int, Union[str, Iterable[Any]]]]):
    
    self.schedules = [from_config(s) for ms, s in schedules]
    milestones = [ms for ms, s in schedules]
    self.milestones = []
    sum_m = 0
    for m in milestones:
        self.milestones.append(sum_m)
        sum_m += m
        

  def get(self, step):
    if isinstance(step, torch.Tensor):
        step = step.item()


    idx = -1 
    for m in self.milestones:
        if m<step:
            idx+=1
        else:
            break
    
    schedule = self.schedules[idx]
    base_idx = self.milestones[idx - 1] if idx >= 1 else 0
    return schedule.get(step - base_idx)


class DelayedSchedule(Schedule):
  """Delays the start of the base schedule."""

  def __init__(self, base_schedule: Schedule, delay_steps, delay_mult):
    self.base_schedule = from_config(base_schedule)
    self.delay_steps = delay_steps
    self.delay_mult = delay_mult

  def get(self, step):
    delay_rate = (self.delay_mult + (1 - self.delay_mult) * math.sin(0.5 * math.pi * min(max(step / self.delay_steps, 0), 1)))

    return delay_rate * self.base_schedule(step)


SCHEDULE_MAP = {
    'constant': ConstantSchedule,
    'linear': LinearSchedule,
    'exponential': ExponentialSchedule,
    'cosine_easing': CosineEasingSchedule,
    'step': StepSchedule,
    'piecewise': PiecewiseSchedule,
    'delayed': DelayedSchedule,
}

