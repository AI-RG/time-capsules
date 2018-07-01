"""Start and train the Capsule Network.

See capsule/trainer.py for flags and more information.
"""

import tensorflow as tf
from capsule import trainer

def main(_):
  trainer.start_and_train()

if __name__ == "__main__":
  tf.app.run()
