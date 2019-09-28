import argparse
import torch
import lib
import numpy as np
import os
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--hidden_size', default=100, type=int) #Literature uses 100 / 1000 --> better is 100
parser.add_argument('--num_layers', default=3, type=int) #1 hidden layer
parser.add_argument('--batch_size', default=50, type=int) #50 in first paper and 32 in second paper
parser.add_argument('--dropout_input', default=0, type=float) #0.5 for TOP and 0.3 for BPR
parser.add_argument('--dropout_hidden', default=0.5, type=float) #0.5 for TOP and 0.3 for BPR
parser.add_argument('--n_epochs', default=5, type=int) #number of epochs (10 in literature)
parser.add_argument('--k_eval', default=20, type=int) #value of K durig Recall and MRR Evaluation
# parse the optimizer arguments
parser.add_argument('--optimizer_type', default='Adagrad', type=str) #Optimizer --> Adagrad is the best according to literature
parser.add_argument('--final_act', default='tanh', type=str) #Final Activation Function
parser.add_argument('--lr', default=0.01, type=float) #learning rate (Best according to literature 0.01 to 0.05)
parser.add_argument('--weight_decay', default=0, type=float) #no weight decay
parser.add_argument('--momentum', default=0, type=float) #no momentum
parser.add_argument('--eps', default=1e-6, type=float) #not used
parser.add_argument("-seed", type=int, default=22, help="Seed for random initialization") #Random seed setting
parser.add_argument("-sigma", type=float, default=None, help="init weight -1: range [-sigma, sigma], -2: range [0, sigma]") # weight initialization [-sigma sigma] in literature

####### TODO: discover this ###########
parser.add_argument("--embedding_dim", type=int, default=-1, help="using embedding") 
####### TODO: discover this ###########

# parse the loss type
parser.add_argument('--loss_type', default='TOP1-max', type=str) #type of loss function TOP1 / BPR / TOP1-max / BPR-max
# etc
parser.add_argument('--time_sort', default=False, type=bool) #In case items are not sorted by time stamp
parser.add_argument('--model_name', default='GRU4REC-CrossEntropy', type=str)
parser.add_argument('--save_dir', default='models', type=str)
parser.add_argument('--data_folder', default='../Dataset/RecSys_Dataset_After/', type=str)
parser.add_argument('--train_data', default='recSys15TrainOnly.txt', type=str)
parser.add_argument('--valid_data', default='recSys15Valid.txt', type=str)
parser.add_argument("--is_eval", action='store_true') #should be used during testing and eliminated during training
parser.add_argument('--load_model', default=None,  type=str)
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')

# Get the arguments
args = parser.parse_args()
args.cuda = torch.cuda.is_available()
#use random seed defined
np.random.seed(args.seed)
torch.manual_seed(args.seed)


if args.cuda:
    torch.cuda.manual_seed(args.seed)

#Write Checkpoints with arguments used in a text file for reproducibility
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

#weight initialization if it was defined
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

    train_data = lib.Dataset(os.path.join(args.data_folder, args.train_data))
    valid_data = lib.Dataset(os.path.join(args.data_folder, args.valid_data), itemmap=train_data.itemmap)
    make_checkpoint_dir()
        
    #set all the parameters according to the defined arguments
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
    #loss function
    loss_function = lib.LossFunction(loss_type=loss_type, use_cuda=args.cuda) #cuda is used with cross entropy only
    if not args.is_eval: #training
        #Initialize the model
        model = lib.GRU4REC(input_size, hidden_size, output_size, final_act=final_act,
                            num_layers=num_layers, use_cuda=args.cuda, batch_size=batch_size,
                            dropout_input=dropout_input, dropout_hidden=dropout_hidden, embedding_dim=embedding_dim)
        #weights initialization
        init_model(model)
        #optimizer
        optimizer = lib.Optimizer(model.parameters(), optimizer_type=optimizer_type, lr=lr,
                                  weight_decay=weight_decay, momentum=momentum, eps=eps)
        #trainer class
        trainer = lib.Trainer(model, train_data=train_data, eval_data=valid_data, optim=optimizer,
                              use_cuda=args.cuda, loss_func=loss_function, batch_size=batch_size, args=args)
        print('#### START TRAINING....')
        trainer.train(0, n_epochs - 1)
    else: #testing
        if args.load_model is not None:
            print("Loading pre-trained model from {}".format(args.load_model))
            try:
                checkpoint = torch.load(args.load_model)
            except:
                checkpoint = torch.load(args.load_model, map_location=lambda storage, loc: storage)
            model = checkpoint["model"]
            model.gru.flatten_parameters()
            evaluation = lib.Evaluation(model, loss_function, use_cuda=args.cuda, k = args.k_eval)
            loss, recall, mrr = evaluation.eval(valid_data, batch_size)
            print("Final result: recall = {:.2f}, mrr = {:.2f}".format(recall, mrr))
        else:
            print("No Pretrained Model was found!")


if __name__ == '__main__':
    main()
