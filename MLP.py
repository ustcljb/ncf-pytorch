import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

import data_util
import evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate")
parser.add_argument("--batch_size", type=int, default=256, help="batch size for training")
parser.add_argument("--epochs", type=int, default=20, help="training epoches")
parser.add_argument("--top_k", type=int, default=10, help="compute metrics@top_k")
parser.add_argument("--embedding_dim", type=int, default=16, help="dimension of embedding")
parser.add_argument("--hidden_layer", type=list, default=[32, 16, 8], help="dimension of each hidden layer")
parser.add_argument("--num_ng", type=int, default=4, help="sample negative items for training")
parser.add_argument("--test_num_ng", type=int, default=99, help="sample part of negative items for testing")
parser.add_argument("--data_set", type=str, default="ml-1m", help="data set. 'ml-1m' or 'pinterest-20'")
parser.add_argument("--data_path", type=str, default="/Users/JingboLiu/Desktop/ncf_pytorch/data")
parser.add_argument("--model_path", type=str, default="/Users/JingboLiu/Desktop/ncf_pytorch/model")
parser.add_argument("--out", default=True, help="save model or not")
parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA")
args = parser.parse_args()

args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

cudnn.benchmark = True


class MLP(nn.Module):
    def __init__(self, user_num, item_num, embedding_dim, hidden_layer, dropout):
        super(MLP, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        embedding_dim: number of embedding dimensions;
        hidden_layer: dimension of each hidden layer (list type);
        dropout: dropout rate between fully connected layers.
        """
        self.dropout = dropout

        self.embed_user = nn.Embedding(user_num, embedding_dim)
        self.embed_item = nn.Embedding(item_num, embedding_dim)

        MLP_modules = []
        self.num_layers = len(hidden_layer)
        for i in range(self.num_layers):
            MLP_modules.append(nn.Dropout(p=self.dropout))
            if i == 0:
                MLP_modules.append(nn.Linear(embedding_dim*2, hidden_layer[0]))
            else:
                MLP_modules.append(nn.Linear(hidden_layer[i-1], hidden_layer[i]))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        self.predict_layer = nn.Linear(hidden_layer[-1], 1)

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=0.01, nonlinearity='leaky_relu')


        # Kaiming/Xavier initialization can not deal with non-zero bias terms
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, user, item):
        embed_user = self.embed_user(user)
        embed_item = self.embed_item(item)
        interaction = torch.cat((embed_user, embed_item), -1)
        output_MLP = self.MLP_layers(interaction)

        output_prediction = self.predict_layer(output_MLP)
        prediction = torch.sigmoid(output_prediction)
        return prediction.view(-1)

if __name__=="__main__":
    data_file = os.path.join(args.data_path, args.data_set)
    train_data, test_data, user_num, item_num, train_mat = data_util.load_all(data_file)

    train_dataset = data_util.NCFData(train_data, item_num, train_mat, args.num_ng, True)
    test_dataset = data_util.NCFData(test_data, item_num, train_mat, 0, False)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = data.DataLoader(test_dataset, batch_size=args.test_num_ng + 1, shuffle=False, num_workers=0)

    model = MLP(user_num, item_num, args.embedding_dim, args.hidden_layer, args.dropout)
    model.to(device=args.device)
    loss_function = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # writer = SummaryWriter() # for visualization

    ########################### TRAINING #####################################
    count, best_hr = 0, 0
    for epoch in range(args.epochs):
        model.train()  # Enable dropout (if have).
        start_time = time.time()
        train_loader.dataset.ng_sample()

        for user, item, label in train_loader:
            user = user.to(device=args.device)
            item = item.to(device=args.device)
            label = label.float().to(device=args.device)

            model.zero_grad()
            prediction = model(user, item)
            loss = loss_function(prediction, label)
            loss.backward()
            optimizer.step()
            # writer.add_scalar('data/loss', loss.item(), count)
            count += 1

        model.eval()
        HR, NDCG = evaluate.metrics(model, test_loader, args.top_k, args.device)

        elapsed_time = time.time() - start_time
        print("The time elapse of epoch {:03d}".format(epoch) + " is: " +
              time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
        print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

        if HR > best_hr:
            best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
            if args.out:
                if not os.path.exists(args.model_path):
                    os.mkdir(args.model_path)
                torch.save(model, os.path.join(args.model_path, 'MLP.pth'))

    print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(best_epoch, best_hr, best_ndcg))