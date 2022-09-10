import torch
import torch.nn as nn
from .embedding import getEmbeddings
from .pcnn import CNNwithPool
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from .Encoder import SAN

class Model(nn.Module):
    ### TODO: improve the BAGATT module
    def __init__(self, word_length, feature_length, cnn_layers, Wv, pf1, pf2, kernel_size,
                 word_size=50, feature_size=5, dropout=0.5, num_classes=53, name='model'):
        super(Model, self).__init__()

        self.word_length = word_length
        self.feature_length = feature_length
        self.cnn_layers = cnn_layers
        self.kernel_size = kernel_size
        self.word_size = word_size
        self.feature_size = feature_size
        self.num_classes = num_classes
        self.name = name
        self.SAN = SAN(self.word_size, 5 , 1)  # 50,5, 1

        self.embeddings = getEmbeddings(self.word_size, self.word_length, self.feature_size,
                                        self.feature_length, Wv, pf1, pf2)
        self.PCNN = CNNwithPool(self.cnn_layers, self.kernel_size)
        self.CNN = nn.Conv2d(1, cnn_layers, kernel_size)

        self.fc1 = nn.Linear(3 * word_size, 3 * word_size)
        self.fc2 = nn.Linear(3 * word_size, 3 * cnn_layers)
        self.dropout = nn.Dropout(dropout)

        self.R_PCNN = nn.Linear(cnn_layers*3, num_classes)
        self.init_linear(self.R_PCNN)

        self.R_CNN = nn.Linear(cnn_layers, num_classes)
        self.init_linear(self.R_CNN)

        self.diag = Variable(torch.ones(self.num_classes).diag().unsqueeze(0)).cuda()

    def init_linear(self, input_linear):
        """
        Initialize linear transformation
        """
        bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
        nn.init.uniform(input_linear.weight, -bias, bias)
        if input_linear.bias is not None:
            input_linear.bias.data.zero_()

    def CNN_ATTBL(self, x, ldist, rdist, pool, total_shape, y_batch):

        embeddings = self.embeddings(x, ldist, rdist)
        cnn, _ = torch.max(self.CNN(embeddings), 2)
        cnn = cnn.squeeze(2)
        batch_sent_emb = self.dropout(cnn)
        batch_sent_emb = nn.functional.tanh(batch_sent_emb)
        batch_e = self.R_CNN(batch_sent_emb)
        batch_p = []
        for i in range(len(total_shape) - 1):
            beg, end = total_shape[i], total_shape[i + 1]
            weight_e = batch_e[beg: end, y_batch[i].cpu().data[0]]
            sent_emb = batch_sent_emb[beg: end]
            alpha = nn.functional.softmax(weight_e, 0)
            s = torch.matmul(alpha, sent_emb)
            o = self.R_CNN(self.dropout(s))
            batch_p.append(o)
        batch_p = torch.stack(batch_p)
        loss = nn.functional.cross_entropy(batch_p, y_batch)
        return loss

    def CNN_ATTRA(self, x, ldist, rdist, pool, total_shape, y_batch):

        embeddings = self.embeddings(x, ldist, rdist)
        cnn, _ = torch.max(self.CNN(embeddings), 2)
        cnn = cnn.squeeze(2)
        batch_sent_emb = self.dropout(cnn)
        batch_sent_emb = nn.functional.tanh(batch_sent_emb)
        batch_e = self.R_CNN(batch_sent_emb)
        batch_score = []
        for i in range(len(total_shape) - 1):
            beg, end = total_shape[i], total_shape[i + 1]
            weight_e = batch_e[beg: end]
            sent_emb = batch_sent_emb[beg: end]
            alpha = nn.functional.softmax(weight_e.transpose(0, 1), 1)
            bag_rep = torch.matmul(alpha, sent_emb)
            o = self.R_CNN(self.dropout(bag_rep))
            batch_score.append(o.diag())
        batch_score = torch.stack(batch_score)
        loss = nn.functional.cross_entropy(batch_score, y_batch)
        return loss

    def CNN_ATTBL_BAGATT(self, x, ldist, rdist, pool, total_shape, y_batch, batch):

        embeddings = self.embeddings(x, ldist, rdist)
        cnn, _ = torch.max(self.CNN(embeddings), 2)
        cnn = cnn.squeeze(2)
        batch_sent_emb = self.dropout(cnn)
        batch_sent_emb = nn.functional.tanh(batch_sent_emb)
        batch_e = self.R_CNN(batch_sent_emb)
        bag_reps = []
        for i in range(len(total_shape) - 1):
            beg, end = total_shape[i], total_shape[i + 1]
            weight_e = batch_e[beg: end, y_batch.view(-1).data[i]]
            sent_emb = batch_sent_emb[beg: end]
            alpha = nn.functional.softmax(weight_e, 0)
            bag_rep = torch.matmul(alpha, sent_emb)
            bag_reps.append(bag_rep)

        bag_reps = torch.stack(bag_reps).view(batch[1],batch[0],self.cnn_layers)
        losses = []
        for i, (bag_rep, y) in enumerate(zip(torch.unbind(bag_reps, 0), torch.unbind(y_batch, 0))):
            if y.data[0] == 0:
                score = self.R_CNN(self.dropout(bag_rep))
                loss = nn.functional.cross_entropy(score, y)
                losses.append(loss)
            else:
                bag_rep = bag_rep / torch.norm(bag_rep, 2, 1, keepdim=True)
                crossatt = torch.matmul(bag_rep, bag_rep.transpose(0,1))
                crossatt = torch.sum(crossatt, 1)
                crossatt = F.softmax(crossatt, 0)
                weighted_bags_rep = torch.matmul(crossatt, bag_rep)
                score = self.R_CNN(self.dropout(weighted_bags_rep)).unsqueeze(0)
                loss = nn.functional.cross_entropy(score, y[0])
                losses.append(loss)
        losses = torch.stack(losses).mean()

        return losses

    def CNN_ATTRA_BAGATT(self, x, ldist, rdist, pool, total_shape, y_batch, batch):

        embeddings = self.embeddings(x, ldist, rdist)
        cnn, _ = torch.max(self.CNN(embeddings), 2)
        cnn = cnn.squeeze(2)
        batch_sent_emb = self.dropout(cnn)
        batch_sent_emb = nn.functional.tanh(batch_sent_emb)
        batch_e = self.R_CNN(batch_sent_emb)
        bag_reps = []
        for i in range(len(total_shape) - 1):
            beg, end = total_shape[i], total_shape[i + 1]
            weight_e = batch_e[beg: end]
            sent_emb = batch_sent_emb[beg: end]
            alpha = nn.functional.softmax(weight_e.transpose(0, 1), 1)
            bag_rep = torch.matmul(alpha, sent_emb)
            bag_reps.append(bag_rep)

        bag_reps = torch.stack(bag_reps).view(batch[1],batch[0],self.num_classes,self.cnn_layers)
        losses = []
        for i, (bag_rep, y) in enumerate(zip(torch.unbind(bag_reps, 0), torch.unbind(y_batch, 0))):
            if y.data[0] == 0:
                score = torch.sum(self.R_CNN(self.dropout(bag_rep)) * self.diag, 2)
                loss = nn.functional.cross_entropy(score, y)
                losses.append(loss)
            else:
                bag_rep = bag_rep.transpose(0,1)
                bag_rep = bag_rep / torch.norm(bag_rep, 2, 2, keepdim=True)
                crossatt = torch.matmul(bag_rep, bag_rep.transpose(1,2))
                crossatt = torch.sum(crossatt, 2)
                crossatt = F.softmax(crossatt, 1)
                weighted_bags_rep = torch.matmul(crossatt.unsqueeze(1), bag_rep).squeeze(1)
                score = self.R_CNN(self.dropout(weighted_bags_rep)).diag().unsqueeze(0)
                loss = nn.functional.cross_entropy(score, y[0])
                losses.append(loss)
        losses = torch.stack(losses).mean()

        return losses

    def decode_CNN(self, x, ldist, rdist, pool, total_shape):

        embeddings = self.embeddings(x, ldist, rdist)
        cnn, _ = torch.max(self.CNN(embeddings), 2)
        cnn = cnn.squeeze(2)
        batch_sent_emb = self.dropout(cnn)
        batch_sent_emb = nn.functional.tanh(batch_sent_emb)
        batch_e = self.R_CNN(batch_sent_emb)
        batch_score = []
        for i in range(len(total_shape) - 1):
            beg, end = total_shape[i], total_shape[i + 1]
            weight_e = batch_e[beg: end]
            sent_emb = batch_sent_emb[beg: end]
            alpha = nn.functional.softmax(weight_e.transpose(0, 1), 1)
            bag_rep = torch.matmul(alpha, sent_emb)
            o = self.R_CNN(self.dropout(bag_rep))
            o = F.softmax(o, 1)
            batch_score.append(o.diag())
        batch_p = torch.stack(batch_score)
        return batch_p

    def PCNN_ATTBL(self, x, ldist, rdist, pool, total_shape, y_batch):

        embeddings = self.embeddings(x, ldist, rdist)
        cnn = self.PCNN(embeddings, pool).view((embeddings.size(0), -1))
        batch_sent_emb = self.dropout(cnn)
        batch_sent_emb = nn.functional.tanh(batch_sent_emb)
        batch_e = self.R_PCNN(batch_sent_emb)
        batch_p = []
        for i in range(len(total_shape) - 1):
            beg, end = total_shape[i], total_shape[i + 1]
            weight_e = batch_e[beg: end, y_batch[i].cpu().data[0]]
            sent_emb = batch_sent_emb[beg: end]
            alpha = nn.functional.softmax(weight_e, 0)
            s = torch.matmul(alpha, sent_emb)
            o = self.R_PCNN(self.dropout(s))
            batch_p.append(o)
        batch_p = torch.stack(batch_p)
        loss = nn.functional.cross_entropy(batch_p, y_batch)
        return loss

    def PCNN_ATTRA(self, x, ldist, rdist, entity_pos, pcnnmask, total_shape, y_batch):      ##通过pcnn模型获取

        leftEnt = []
        rightEnt = []
        for i in range(len(entity_pos)):
            leftEnt.append(x[i][entity_pos[i][0] + 1])
            rightEnt.append(x[i][entity_pos[i][1] + 1])
        leftEnt = torch.stack(leftEnt)
        rightEnt = torch.stack(rightEnt)
        # embeddings = self.embeddings(x, ldist, rdist, leftEnt, rightEnt)    ## 使用embedding对sequence，pf1和pf2进行嵌入式表
        Xp, Xe = self.embeddings(x, ldist, rdist, leftEnt, rightEnt)    ## 使用embedding对sequence，pf1和pf2进行嵌入式表

        batch_sent_pcnn = self.PCNN(Xp, pcnnmask).view((Xp.size(0), -1)) # [sentence num, 690]
        batch_entity = self.SAN(Xp.squeeze(1), Xe)

        # Combine
        batch_sent_emb = self.selective_gate(batch_sent_pcnn, batch_entity)

        batch_sent_emb = self.dropout(batch_sent_emb)
        batch_sent_emb = nn.functional.tanh(batch_sent_emb)
        batch_e = self.R_PCNN(batch_sent_emb)   # torch.Size() [句子长度, 关系数量(53)]
        batch_score = []
        for i in range(len(total_shape) - 1):   ## Intra-Bag Attention, i+1则为一个包
            beg, end = total_shape[i], total_shape[i + 1]
            weight_e = batch_e[beg: end]    ## 获取[1, 53]
            sent_emb = batch_sent_emb[beg: end]
            alpha = nn.functional.softmax(weight_e.transpose(0, 1), 1)      ## alpha : ai[53,1]
            bag_rep = torch.matmul(alpha, sent_emb) # alpha:[53,1], sent_emb:[1, 690] bag_rep:bi
            o = self.R_PCNN(self.dropout(bag_rep))  ## [53, 53]
            batch_score.append(o.diag())
        batch_score = torch.stack(batch_score)  # [50,53]
        loss = nn.functional.cross_entropy(batch_score, y_batch)    # 0.673, 0.456
        return loss ### [1]

    def PCNN_ATTBL_BAGATT(self, x, ldist, rdist, pool, total_shape, y_batch, batch):

        embeddings = self.embeddings(x, ldist, rdist)
        batch_sent_emb = self.PCNN(embeddings, pool).view((embeddings.size(0), -1))
        batch_sent_emb = self.dropout(batch_sent_emb)
        batch_sent_emb = nn.functional.tanh(batch_sent_emb)
        batch_e = self.R_PCNN(batch_sent_emb)
        bag_reps = []
        for i in range(len(total_shape) - 1):
            beg, end = total_shape[i], total_shape[i + 1]
            weight_e = batch_e[beg: end, y_batch.view(-1).data[i]]
            sent_emb = batch_sent_emb[beg: end]
            alpha = nn.functional.softmax(weight_e, 0)
            bag_rep = torch.matmul(alpha, sent_emb)
            bag_reps.append(bag_rep)

        bag_reps = torch.stack(bag_reps).view(batch[1],batch[0],self.cnn_layers*3)
        losses = []
        for i, (bag_rep, y) in enumerate(zip(torch.unbind(bag_reps, 0), torch.unbind(y_batch, 0))):
            if y.data[0] == 0:
                score = self.R_PCNN(self.dropout(bag_rep))
                loss = nn.functional.cross_entropy(score, y)
                losses.append(loss)
            else:
                bag_rep = bag_rep / torch.norm(bag_rep, 2, 1, keepdim=True)
                crossatt = torch.matmul(bag_rep, bag_rep.transpose(0,1))
                crossatt = torch.sum(crossatt, 1)
                crossatt = F.softmax(crossatt, 0)
                weighted_bags_rep = torch.matmul(crossatt, bag_rep)
                score = self.R_PCNN(self.dropout(weighted_bags_rep)).unsqueeze(0)
                loss = nn.functional.cross_entropy(score, y[0])
                losses.append(loss)
        losses = torch.stack(losses).mean()
        return losses

    def PCNN_ATTRA_BAGATT(self, x, ldist, rdist, pcnnmask, total_shape, y_batch, batch):

        embeddings = self.embeddings(x, ldist, rdist)   ## [88, 1, 82, 60]
        batch_sent_emb = self.PCNN(embeddings, pcnnmask).view((embeddings.size(0), -1)) ## pcnn利用mask进行训练
        batch_sent_emb = self.dropout(batch_sent_emb)
        batch_sent_emb = nn.functional.tanh(batch_sent_emb)
        batch_e = self.R_PCNN(batch_sent_emb)   # [sequence num, rel nums]
        bag_reps = []
        for i in range(len(total_shape) - 1):
            beg, end = total_shape[i], total_shape[i + 1]
            weight_e = batch_e[beg: end]
            sent_emb = batch_sent_emb[beg: end]
            alpha = nn.functional.softmax(weight_e.transpose(0, 1), 1)
            bag_rep = torch.matmul(alpha, sent_emb)
            bag_reps.append(bag_rep)

        bag_reps = torch.stack(bag_reps).view(batch[1],batch[0],self.num_classes,self.cnn_layers*3)
        losses = []
        for i, (bag_rep, y) in enumerate(zip(torch.unbind(bag_reps, 0), torch.unbind(y_batch, 0))): ##[5, rel nums, 230*3]
            if y.data[0] == 0:
                score = torch.sum(self.R_PCNN(self.dropout(bag_rep)) * self.diag, 2)    ## bag_rep:[5, 53, 690] -> [5, 53]
                loss = nn.functional.cross_entropy(score, y)    # y: 5
                losses.append(loss)
            else:       # ##如果是其他关系则选取其中最有代表的句子进行计算
                bag_rep = bag_rep.transpose(0,1)    ## bag_req:[sen nums of bag, rel nums, 690]
                bag_rep = bag_rep / torch.norm(bag_rep, 2, 2, keepdim=True)
                crossatt = torch.matmul(bag_rep, bag_rep.transpose(1,2))    ## 计算bik与本身的similarity
                crossatt = torch.sum(crossatt, 2)       ## [53, 5, 5]
                crossatt = F.softmax(crossatt, 1)   ## 53,5
                weighted_bags_rep = torch.matmul(crossatt.unsqueeze(1), bag_rep).squeeze(1) ## bag_rep : [5, 53, 690] -> [53, 690],求得表示包组的向量gk
                score = self.R_PCNN(self.dropout(weighted_bags_rep)).diag().unsqueeze(0)
                loss = nn.functional.cross_entropy(score, y[0:1])
                losses.append(loss)
        losses = torch.stack(losses).mean()

        return losses

    def PCNN_ATTRA_BAGATT2(self, x, ldist, rdist, entity_pos, pcnnmask, total_shape, y_batch, batch):

        leftEnt = []
        rightEnt = []
        for i in range(len(entity_pos)):
            leftEnt.append(x[i][entity_pos[i][0] + 1])
            rightEnt.append(x[i][entity_pos[i][1] + 1])
        leftEnt = torch.stack(leftEnt)
        rightEnt = torch.stack(rightEnt)
        # embeddings = self.embeddings(x, ldist, rdist, leftEnt, rightEnt)    ## 使用embedding对sequence，pf1和pf2进行嵌入式表
        Xp, Xe = self.embeddings(x, ldist, rdist, leftEnt, rightEnt)  ## 使用embedding对sequence，pf1和pf2进行嵌入式表

        batch_sent_pcnn = self.PCNN(Xp, pcnnmask).view((Xp.size(0), -1))  # [sentence num, 690]
        batch_entity = self.SAN(Xp.squeeze(1), Xe)

        # Combine
        batch_sent_emb = self.selective_gate(batch_sent_pcnn, batch_entity)

        batch_sent_emb = self.dropout(batch_sent_emb)
        batch_sent_emb = nn.functional.tanh(batch_sent_emb)
        batch_e = self.R_PCNN(batch_sent_emb)  # torch.Size() [句子长度, 关系数量(53)]

        # embeddings = self.embeddings(x, ldist, rdist, leftEnt, rightEnt)
        # batch_sent_emb = self.PCNN(embeddings, pcnnmask).view((embeddings.size(0), -1)) ## pcnn利用mask进行训练
        # batch_sent_emb = self.dropout(batch_sent_emb)
        # batch_sent_emb = nn.functional.tanh(batch_sent_emb)
        # batch_e = self.R_PCNN(batch_sent_emb)   # [sequence num, rel nums]
        bag_reps = []
        for i in range(len(total_shape) - 1):
            beg, end = total_shape[i], total_shape[i + 1]
            weight_e = batch_e[beg: end]
            sent_emb = batch_sent_emb[beg: end]
            alpha = nn.functional.softmax(weight_e.transpose(0, 1), 1)
            bag_rep = torch.matmul(alpha, sent_emb)
            bag_reps.append(bag_rep)

        bag_reps = torch.stack(bag_reps).view(batch[1],batch[0],self.num_classes,self.cnn_layers*3)
        losses = []
        for i, (bag_rep, y) in enumerate(zip(torch.unbind(bag_reps, 0), torch.unbind(y_batch, 0))): ##[5, rel nums, 230*3]
            if y.data[0] == 0:
                score = torch.sum(self.R_PCNN(self.dropout(bag_rep)) * self.diag, 2)    ## bag_rep:[5, 53, 690] -> [5, 53]
                loss = nn.functional.cross_entropy(score, y)    # y: 5
                losses.append(loss)
            else:       # ##如果是其他关系则选取其中最有代表的句子进行计算
                bag_rep = bag_rep.transpose(0,1)    ## bag_req:[sen nums of bag, rel nums, 690]
                bag_rep = bag_rep / torch.norm(bag_rep, 2, 2, keepdim=True)
                crossatt = torch.matmul(bag_rep, bag_rep.transpose(1,2))    ## 计算bik与本身的similarity
                crossatt = torch.sum(crossatt, 2)       ## [53, 5, 5]
                crossatt = F.softmax(crossatt, 1)   ## 53,5
                weighted_bags_rep = torch.matmul(crossatt.unsqueeze(1), bag_rep).squeeze(1) ## bag_rep : [5, 53, 690] -> [53, 690],求得表示包组的向量gk
                score = self.R_PCNN(self.dropout(weighted_bags_rep)).diag().unsqueeze(0)
                loss = nn.functional.cross_entropy(score, y[0:1])
                losses.append(loss)
        losses = torch.stack(losses).mean()

        return losses

    def decode_PCNN(self, x, ldist, rdist, entity_pos, pcnnmask, total_shape):

        # embeddings = self.embeddings(x, ldist, rdist)
        leftEnt = []
        rightEnt = []
        for i in range(len(entity_pos)):
            leftEnt.append(x[i][entity_pos[i][0] + 1])
            rightEnt.append(x[i][entity_pos[i][1] + 1])
        leftEnt = torch.stack(leftEnt)
        rightEnt = torch.stack(rightEnt)
        # embeddings = self.embeddings(x, ldist, rdist, leftEnt, rightEnt)    ## 使用embedding对sequence，pf1和pf2进行嵌入式表
        Xp, Xe = self.embeddings(x, ldist, rdist, leftEnt, rightEnt)

        batch_sent_pcnn = self.PCNN(Xp, pcnnmask).view((Xp.size(0), -1))  # [sentence num, 690]
        batch_entity = self.SAN(Xp.squeeze(1), Xe)

        # Combine
        batch_sent_emb = self.selective_gate(batch_sent_pcnn, batch_entity)

        batch_sent_emb = self.dropout(batch_sent_emb)
        batch_sent_emb = nn.functional.tanh(batch_sent_emb)
        batch_e = self.R_PCNN(batch_sent_emb)  # torch.Size() [句子长度, 关系数量(53)]
        # batch_score = []
        #
        # cnn = self.PCNN(embeddings, pcnnmask).view((embeddings.size(0), -1))
        # batch_sent_emb = self.dropout(cnn)
        # batch_sent_emb = nn.functional.tanh(batch_sent_emb)
        # batch_e = self.R_PCNN(batch_sent_emb)
        batch_score = []
        for i in range(len(total_shape) - 1):   ## 计算每个包的表示bi
            beg, end = total_shape[i], total_shape[i + 1]
            weight_e = batch_e[beg: end]
            sent_emb = batch_sent_emb[beg: end]
            alpha = nn.functional.softmax(weight_e.transpose(0, 1), 1)
            bag_rep = torch.matmul(alpha, sent_emb)
            o = self.R_PCNN(self.dropout(bag_rep))
            o = F.softmax(o, 1)
            batch_score.append(o.diag())
        batch_p = torch.stack(batch_score)  # [500, 53]
        return batch_p

    def forward(self, x, ldist, rdist, pool, total_shape, y_batch):
        pass

    def selective_gate(self, S, U):
        G = torch.sigmoid(self.fc2(torch.tanh(self.fc1(U))))
        X = G * S
        # B = []  # Bag Output
        # for s in X_Scope:
        #     B.append(X[s[0]:s[1]].mean(0))
        # B = torch.stack(B)
        return X
