import argparse
import platform

if platform.system() == "Windows":
    DEVICE = "cuda:0"

elif platform.system() == "Linux":
    DEVICE = "cuda:0"




elif platform.system() == "Darwin":
    DEVICE = "mps:0"

else:
    raise ValueError("Unknown platform: {}".format(platform.system()))

parser = argparse.ArgumentParser(
    description='Model for determining whether two faces belong to same identity')
parser.add_argument('--eval', action='store_true', default=False,
                    help='whether to only run evaluation')
parser.add_argument('--device', default=DEVICE, type=
str, help='device to run the experiments')
parser.add_argument('--use-test', action='store_true', default=False,
                    help='Use the test set or not')
parser.add_argument('--disable-distractor', action='store_true', default=False,
                    help='Whether or not to disable distractors')
parser.add_argument('--label-ratio', type=float, default=0.1, metavar='N',
                    help='Portion of labeled images in the training set')
parser.add_argument('--mode-ratio', type=float, default=1.0, metavar='N',
                    help='Portion of modes in the training set')
parser.add_argument('--nclasses-eval', type=int, default=5, metavar='N',
                    help='Number of classes for testing')
parser.add_argument('--nclasses-train', type=int, default=20, metavar='N',
                    help='Number of classes in an update')
parser.add_argument('--nsuperclassestrain', type=int, default=-1, metavar='N',
                    help='Number of superclasses in an episode')
parser.add_argument('--nsuperclasseseval', type=int, default=-1, metavar='N',
                    help='Number of superclasses for testing')
parser.add_argument('--nclasses-episode', type=int, default=5, metavar='N',
                    help='Number of classes in an episode')
parser.add_argument('--accumulation-steps', type=int, default=1, metavar='N',
                    help='Number of accumulation steps for an update')
parser.add_argument('--nshot', type=int, default=1, metavar='N', help='nshot')
parser.add_argument('--num-eval-episode', type=int, default=500, metavar='N',
                    help='Number of evaluation episodes')
parser.add_argument('--num-test', type=int, default=-1, metavar='N',
                    help='Number of test images per episode')
parser.add_argument('--num-unlabel', type=int, default=5, metavar='N',
                    help='Number of unlabeled for training')
parser.add_argument('--num-unlabel-test', type=int, default=5, metavar='N',
                    help='Number of unlabeled for testing')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--data-root', default=None, help='Data root')
parser.add_argument('--dataset', default="omniglot", help='Dataset name')
parser.add_argument('--model', default="basic", help='Model name')
parser.add_argument('--results', default='./results',
                    help='Checkpoint save path')
parser.add_argument('--super-classes', action='store_true', default=False,
                    help='Use super-class labels')
parser.add_argument('--pretrain', default=None,
                    help='folder of the model to load')
args = parser.parse_args()
