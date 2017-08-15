import tensorflow as tf
import flags

def main(_):
    flags.flag_test()

if __name__ == "__main__":
    tf.app.run() #initialize flags