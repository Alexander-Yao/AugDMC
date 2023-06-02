import argparse


def get_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default="fruit", help='dataset')
    parser.add_argument('--batch_size', type=int, default=10, help='training batch size')
    parser.add_argument('--tau', type=float, default=1.0, help='softmax tau')
    parser.add_argument('--seed', type=int, default=1, help='seed')
    parser.add_argument('--epoch', type=int, default=1000, help='the number of epochs')
    parser.add_argument('--type', type=str, default='tmp', help='')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--save', type=bool, default=False, help='save model')
    args = parser.parse_args()
    return args
