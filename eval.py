import os
import paddle
import paddle.optimizer as optim
from paddle.io import DataLoader
from model import TMC
from data import Multi_view_data
import warnings
import time
import sys

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

paddle.seed(42)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                        help='input batch size for training [default: 100]')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--lambda-epochs', type=int, default=50, metavar='N',
                        help='gradually increase the value of lambda from 0 to 1')
    parser.add_argument('--lr', type=float, default=0.003, metavar='LR',
                        help='learning rate')
    args = parser.parse_args()
    args.data_name = 'handwritten_6views'
    args.data_path = 'datasets/' + args.data_name
    args.dims = [[240], [76], [216], [47], [64], [6]]
    args.views = len(args.dims)
    '''
    train_loader = torch.utils.data.DataLoader(
        Multi_view_data(args.data_path, train=True), batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        Multi_view_data(args.data_path, train=False), batch_size=args.batch_size, shuffle=False)
    '''
    train_loader = paddle.io.DataLoader(
        Multi_view_data(args.data_path, train=True), batch_size=args.batch_size, shuffle=True)
    test_loader = paddle.io.DataLoader(
        Multi_view_data(args.data_path, train=False), batch_size=args.batch_size, shuffle=False)

    N_mini_batches = len(train_loader)
    print('The number of training images = %d' % N_mini_batches)

    model = TMC(10, args.views, args.dims, args.lambda_epochs)
    optimizer = optim.Adam(parameters=model.parameters(), learning_rate=args.lr, weight_decay=1e-4)

    #model.cuda()
    paddle.device.set_device("gpu:0")
    

    def test(epoch):
        model.eval()
        loss_meter = AverageMeter()
        correct_num, data_num = 0, 0
        for batch_idx, (data, target) in enumerate(test_loader):
            for v_num in range(len(data)):
                data[v_num] = paddle.to_tensor(data[v_num],dtype='float32',place=paddle.CPUPlace())
            data_num += target.shape[0]

            with paddle.no_grad():
                target = paddle.to_tensor(target,dtype='int64',place=paddle.CPUPlace())
                evidences, evidence_a, loss = model(data, target, epoch)

                predicted = paddle.argmax(evidence_a, axis=1)
                correct_num += (predicted == target).sum().item()
                loss_meter.update(loss.item())

            break



        #print('epoch:',epoch,'====> acc: {:.4f}'.format(correct_num/data_num))
        return loss_meter.avg, correct_num/data_num

    # load
    layer_state_dict = paddle.load("output/model_best/model_last.pdparams")
    opt_state_dict = paddle.load("output/model_best/model_last.pdopt")

    model.set_state_dict(layer_state_dict)
    optimizer.set_state_dict(opt_state_dict)    

    epoch=0
    test_loss, acc = test(epoch)
    print('loss:',test_loss,'acc: {:.4f}'.format(acc))
