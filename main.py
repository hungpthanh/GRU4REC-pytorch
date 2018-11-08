import argparse
import torch
import lib
import numpy as np
import os
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--hidden_size', default=100, type=int)
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--batch_size', default=50, type=int)
parser.add_argument('--dropout_input', default=0, type=float)
parser.add_argument('--dropout_hidden', default=.5, type=float)

# parse the optimizer arguments
parser.add_argument('--optimizer_type', default='Adagrad', type=str)
parser.add_argument('--final_act', default='tanh', type=str)
parser.add_argument('--lr', default=.01, type=float)
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--momentum', default=0, type=float)
parser.add_argument('--eps', default=1e-6, type=float)

parser.add_argument("-seed", type=int, default=7,
                     help="Seed for random initialization")
parser.add_argument("-sigma", type=float, default=None,
                     help="init weight -1: range [-sigma, sigma], -2: range [0, sigma]")
parser.add_argument("--embedding_dim", type=int, default=-1,
                     help="using embedding")
# parse the loss type
parser.add_argument('--loss_type', default='TOP1', type=str)

# etc
parser.add_argument('--n_epochs', default=5, type=int)
parser.add_argument('--time_sort', default=False, type=bool)
parser.add_argument('--model_name', default='GRU4REC', type=str)
parser.add_argument('--save_dir', default='models', type=str)
parser.add_argument('--data_folder', default='data/preprocessed_data', type=str)
parser.add_argument('--train_data', default='rsc15_train_full.txt', type=str)
parser.add_argument('--valid_data', default='rsc15_test.txt', type=str)
parser.add_argument('--test_data', default='rsc15_test.txt', type=str)
parser.add_argument("--is_eval", action='store_true')
parser.add_argument('--load_model', default=None,  type=str)
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
# Get the arguments
args = parser.parse_args()
args.cuda = torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(7)


if args.cuda:
    torch.cuda.manual_seed(args.seed)


def make_checkpoint_dir():
    print("PARAMETER" + "-"*10)
    now = datetime.datetime.now()
    S = '{:02d}{:02d}{:02d}{:02d}'.format(now.month, now.day, now.hour, now.minute)
    save_dir = os.path.join(args.checkpoint_dir, S)
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    args.checkpoint_dir = save_dir

    with open(os.path.join(args.checkpoint_dir, 'parameter.txt'), 'w') as f:
        for attr, value in sorted(args.__dict__.items()):
            print("{}={}".format(attr.upper(), value))
            f.write("{}={}\n".format(attr.upper(), value))

    print("---------" + "-"*10)


def init_model(model):
    if args.sigma is not None:
        for p in model.parameters():
            if args.sigma != -1 and args.sigma != -2:
                sigma = args.sigma
                p.data.uniform_(-sigma, sigma)
            elif len(list(p.size())) > 1:
                sigma = np.sqrt(6.0 / (p.size(0) + p.size(1)))
                if args.sigma == -1:
                    p.data.uniform_(-sigma, sigma)
                else:
                    p.data.uniform_(0, sigma)


def main():
    print("Loading train data from {}".format(os.path.join(args.data_folder, args.train_data)))
    print("Loading valid data from {}".format(os.path.join(args.data_folder, args.valid_data)))
    print("Loading test data from {}\n".format(os.path.join(args.data_folder, args.test_data)))

    train_data = lib.Dataset(os.path.join(args.data_folder, args.train_data))
    valid_data = lib.Dataset(os.path.join(args.data_folder, args.valid_data), itemmap=train_data.itemmap)
    test_data = lib.Dataset(os.path.join(args.data_folder, args.test_data))

    if not args.is_eval:
        make_checkpoint_dir()

    input_size = len(train_data.items)
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    output_size = input_size
    batch_size = args.batch_size
    dropout_input = args.dropout_input
    dropout_hidden = args.dropout_hidden
    embedding_dim = args.embedding_dim
    final_act = args.final_act
    loss_type = args.loss_type

    optimizer_type = args.optimizer_type
    lr = args.lr
    weight_decay = args.weight_decay
    momentum = args.momentum
    eps = args.eps

    n_epochs = args.n_epochs
    time_sort = args.time_sort

    if not args.is_eval:
        model = lib.GRU4REC(input_size, hidden_size, output_size,
                            final_act=final_act,
                            num_layers=num_layers,
                            use_cuda=args.cuda,
                            batch_size=batch_size,
                            dropout_input=dropout_input,
                            dropout_hidden=dropout_hidden,
                            embedding_dim=embedding_dim
                            )

        # init weight
        # See Balazs Hihasi(ICLR 2016), pg.7

        init_model(model)

        optimizer = lib.Optimizer(model.parameters(),
                                  optimizer_type=optimizer_type,
                                  lr=lr,
                                  weight_decay=weight_decay,
                                  momentum=momentum,
                                  eps=eps)

        loss_function = lib.LossFunction(loss_type=loss_type, use_cuda=args.cuda)

        trainer = lib.Trainer(model,
                              train_data=train_data,
                              eval_data=valid_data,
                              optim=optimizer,
                              use_cuda=args.cuda,
                              loss_func=loss_function,
                              args=args)

        trainer.train(0, n_epochs - 1)
    else:
        if args.load_model is not None:
            print("Loading pre trained model from {}".format(args.load_model))
            checkpoint = torch.load(args.load_model)
            model = checkpoint["model"]
            model.gru.flatten_parameters()
            optim = checkpoint["optim"]
            loss_function = lib.LossFunction(loss_type=loss_type, use_cuda=args.cuda)
            evaluation = lib.Evaluation(model, loss_function, use_cuda=args.cuda)
            loss, recall, mrr = evaluation.eval(valid_data)
            print("Final result: recall = {:.2f}, mrr = {:.2f}".format(recall, mrr))
        else:
            print("Pre trained model is None!")


if __name__ == '__main__':
    main()
