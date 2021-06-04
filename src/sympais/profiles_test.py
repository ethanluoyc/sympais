import unittest

from numpyro import distributions

from sympais import profiles


class ProfileTest(unittest.TestCase):

  def test_discrete_triggers_warnings(self):
    with self.assertWarns(Warning):
      profiles.IndependentProfile({'x': distributions.Categorical(probs=[0, 1.])})


if __name__ == '__main__':
  unittest.main()
