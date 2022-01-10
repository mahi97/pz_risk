import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--hidden-size',
        type=int,
        default=20,
        help='Size of hidden feature in NN (default: 20)')
    parser.add_argument(
        '--max-agents',
        type=int,
        default=3,
        help='Maximum number of agents (default: 6)')
    parser.add_argument(
        '--max-nodes',
        type=int,
        default=10,
        help='Maximum number of nodes (default: 20)')
    parser.add_argument(
        '--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--edge-p',
        type=float,
        default=0.5,
        help='Edge existence probability in random graph generation (default: 0.5)')
    parser.add_argument(
        '--unit-coef',
        type=float,
        default=2.0,
        help='Unit Coefficient (default: 2.0)')

    parser.add_argument(
        '--mcts', type=int, default=100, help='MCTS Rollouts (default: 100)')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--max_episode',
        type=int,
        default=300,
        help='Number of matches to play in total (default: 3000)')
    parser.add_argument(
        '--load',
        type=int,
        default=0,
        help='Last checkpoint to load (default: 0)')
    parser.add_argument(
        '--num-processes',
        type=int,
        default=1,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=300,
        help='number steps in every episode (default: 300)')
    parser.add_argument(
        '--train-epoch',
        type=int,
        default=10,
        help='number of critic train epochs (default: 100)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=10,
        help='save interval, one save per n updates (default: 10)')
    parser.add_argument(
        '--eval-iter',
        type=int,
        default=5,
        help='eval iteration to decide acceptance of new model (default: 16)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=None,
        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=10e6,
        help='number of environment steps to train (default: 10e6)')
    parser.add_argument(
        '--env-name',
        default='Risk-Normal-6-v0',
        help='environment to train on (default: Risk-Normal-6-v0)')
    parser.add_argument(
        '--log-dir',
        default='/tmp/gym/',
        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument(
        '--save-path',
        default='./trained_models/',
        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=False,
        help='compute returns taking into account time limits')
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=False,
        help='use a linear schedule on the learning rate')
    parser.add_argument(
        '--dir',
        default='./mini_7/100.pt',
        help='Directory to load')



    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args
