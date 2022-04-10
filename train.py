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
    def train(epoch):
        model.train()
        # training log
        train_reader_cost = 0.0
        train_run_cost = 0.0
        total_samples = 0
        acc = 0.0
        reader_start = time.time()
        batch_past = 0
        print_freq=4

        loss_meter = AverageMeter()
        correct_num, data_num = 0, 0
        for batch_idx, (data, target) in enumerate(train_loader):
            train_reader_cost += time.time() - reader_start
            train_start = time.time()    

            for v_num in range(len(data)):
                data[v_num] = paddle.to_tensor(data[v_num],dtype='float32',place=paddle.CPUPlace())
            data_num += target.shape[0]
            target = paddle.to_tensor(target,dtype='int64',place=paddle.CPUPlace())
            # refresh the optimizer

            optimizer.clear_grad()
            train_run_cost += time.time() - train_start
            total_samples = data_num

            batch_past += 1
            evidences, evidence_a, loss = model(data, target, epoch)
            # compute gradients and take step
            predicted = paddle.argmax(evidence_a, axis=1)
            correct_num += (predicted == target).sum().item()

            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())


            acc=correct_num/data_num
            if epoch%50==0 and batch_idx > 0 and (batch_idx+1) % print_freq == 0:
                msg = "[Epoch {}, iter: {}] acc: {:.5f}, lr: {:.5f}, loss: {:.5f}, avg_reader_cost: {:.5f} sec, avg_batch_cost: {:.5f} sec, avg_samples: {}, avg_ips: {:.5f} images/sec.".format(
                    epoch, batch_idx, acc / batch_past,
                    optimizer.get_lr(),
                    loss.item(), train_reader_cost / batch_past,
                    (train_reader_cost + train_run_cost) / batch_past,
                    total_samples / batch_past,
                    total_samples / (train_reader_cost + train_run_cost))
                # just log on 1st device
                if paddle.distributed.get_rank() <= 0:
                    print(msg)
            if epoch%100==0 and batch_idx > 0 and (batch_idx+1) % 8 == 0:
                test_loss, acc1 = test(epoch)
                print('epoch:',epoch,',test_loss:{:.5f},'.format(test_loss),'test_acc: {:.4f}'.format(acc1))

            sys.stdout.flush()
            train_reader_cost = 0.0
            train_run_cost = 0.0
            total_samples = 0
            acc = 0.0
            batch_past = 0

            reader_start = time.time()


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



        #print('epoch:',epoch,'====> acc: {:.4f}'.format(correct_num/data_num))
        return loss_meter.avg, correct_num/data_num

    for epoch in range(1, args.epochs + 1):
        train(epoch)

    test_loss, acc = test(epoch)
    print('====> acc: {:.4f}'.format(acc))
