import argparse
import torch
import lib

parser = argparse.ArgumentParser()
parser.add_argument('--hidden_size', default=100, type=int)
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--batch_size', default=50, type=int)
parser.add_argument('--dropout_input', default=0, type=float)
parser.add_argument('--dropout_hidden', default=.5, type=float)

# parse the optimizer arguments
parser.add_argument('--optimizer_type', default='Adagrad', type=str)
parser.add_argument('--lr', default=.01, type=float)
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--momentum', default=0, type=float)
parser.add_argument('--eps', default=1e-6, type=float)

# parse the loss type
parser.add_argument('--loss_type', default='TOP1', type=str)

# etc
parser.add_argument('--n_epochs', default=5, type=int)
parser.add_argument('--time_sort', default=False, type=bool)
parser.add_argument('--model_name', default='GRU4REC', type=str)
parser.add_argument('--save_dir', default='models', type=str)

# Get the arguments
args = parser.parse_args()
args.cuda = torch.cuda.is_available()


def main():
    train_data = lib.Dataset('data/preprocessed_data/rsc15_train_full.txt')
    valid_data = lib.Dataset('data/preprocessed_data/rsc15_test.txt')
    test_data = lib.Dataset('data/preprocessed_data/rsc15_test.txt')

    input_size = len(train_data.items)
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    output_size = input_size
    batch_size = args.batch_size
    dropout_input = args.dropout_input
    dropout_hidden = args.dropout_hidden

    loss_type = args.loss_type

    optimizer_type = args.optimizer_type
    lr = args.lr
    weight_decay = args.weight_decay
    momentum = args.momentum
    eps = args.eps

    n_epochs = args.n_epochs
    time_sort = args.time_sort
    torch.manual_seed(7)

    model = lib.GRU4REC(input_size, hidden_size, output_size,
                        num_layers=num_layers,
                        use_cuda=args.cuda,
                        batch_size=batch_size,
                        dropout_input=dropout_input,
                        dropout_hidden=dropout_hidden
                        )

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

    trainer.train(0, n_epochs)


if __name__ == '__main__':
    main()
