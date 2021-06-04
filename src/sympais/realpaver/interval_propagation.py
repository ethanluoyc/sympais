import collections
import re
import subprocess
import tempfile
from typing import List

from sympais.realpaver.realpaver_api import RealPaverInput

_constraint_pat = r"([a-zA-Z\d_]+)\s+in\s+\[(.*),(.*)\]"
_box_name_pat = r"(INNER|OUTER)\sBOX\s(\d+)"


class _Interval(collections.namedtuple("Interval", "lower, upper")):
  pass


class Box:
  """Represent a box returned from RealPaver"""

  def __init__(self, box_id, box_type):
    self.box_id = box_id
    self.box_type = box_type
    self.intervals = collections.OrderedDict()

  def add_interval(self, variable, lower, upper):
    self.intervals[variable] = _Interval(lower, upper)

  def __repr__(self):
    out = ""
    out += "{} BOX {}".format(self.box_type, self.box_id)
    out += "\n"
    for var, interval in self.intervals.items():
      out += "{} in [{} , {}]\n".format(var, interval.lower, interval.upper)
    return out


class BoxList(collections.UserList):

  def __init__(self, boxes: List[Box]):
    boxes = [b for b in boxes if b.box_type != "INITIAL"]
    super().__init__(boxes)

  @property
  def inner_boxes(self):
    return [b for b in self.data if b.box_type == "INNER"]

  @property
  def outer_boxes(self):
    return [b for b in self.data if b.box_type == "OUTER"]


def find_boxes(input_, n_boxes=None, precision=None, bin_path="./bin/realpaver"):
  if isinstance(input_, RealPaverInput):
    input_ = input_.render()
  with tempfile.NamedTemporaryFile("wt") as file:
    file.write(input_)
    file.flush()
    # TODO
    if n_boxes is not None:
      cmd = [bin_path, "-bound", "-paving", "-number", str(n_boxes), file.name]
    elif precision is not None:
      cmd = [bin_path, "-bound", "-paving", "-precision", str(precision), file.name]
    else:
      raise ValueError("Either n_boxes or precision needs to be used.")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return _parse_boxes(proc.stdout)


def _parse_box_name(s: str):
  matched = re.search(_box_name_pat, s)
  if matched:
    inner_outer = matched.group(1)
    box_id = matched.group(2)
    return inner_outer, int(box_id)
  else:
    return None


def _parse_boxes(output: str):
  boxes = []
  current_box = None
  for line in output.splitlines():
    if "BOX" in line:
      if "INITIAL" in line:
        box_type, box_id = ("INITIAL", "")
      else:
        box_type, box_id = _parse_box_name(line)  # pytype: disable=attribute-error
      new_box = Box(box_id=box_id, box_type=box_type)
      boxes.append(new_box)
      current_box = new_box
    matched = re.search(_constraint_pat, line)
    if matched:
      current_box.add_interval(  # pytype: disable=attribute-error
          matched.group(1), float(matched.group(2)), float(matched.group(3)))
  return BoxList(boxes)
