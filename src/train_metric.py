from PIL import Image
from argparse import ArgumentParser
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import SGD, Adam
from tensorboardX import SummaryWriter
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import ModelCheckpoint
from ignite.engine.engine import Engine
from ignite._utils import convert_tensor

from loss.triplet_loss import TripletLoss
from model.siamese import FeatureExtractor
from online_mining_dataset import OnlineMiningDataset
from model.debug_model import DebugModel
#from metrics import TripletAccuracy, TripletLoss
from transforms import get_train_transform
from torchvision.datasets import MNIST


def get_data_loaders(train_batch_size, prob):

    train_data_transform = get_train_transform()
    if args.dataset == 'whale':
        train_loader = DataLoader(OnlineMiningDataset('data', transform=train_data_transform, min_size=args.min_size_per_class),
                                  batch_size=train_batch_size, shuffle=False)
    elif args.dataset == 'mnist':
        print('mnistで試します')
        data = MNIST('~/.pytorch/mnist', download=True,
                     transform=train_data_transform)
        train_loader = DataLoader(
            data, batch_size=train_batch_size, shuffle=True)
    return train_loader


def create_summary_writer(model, data_loader, log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    data_loader_iter = iter(data_loader)
    batch = next(data_loader_iter)
    try:
        writer.add_graph(model, *batch)
    except Exception as e:
        print("Failed to save model graph: {}".format(e))
    return writer


def create_triplet_trainer(model, optimizer, loss_fn, device=None):
    if device:
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        a, p, n = map(lambda x: convert_tensor(x, device), batch)
        a, p, n = model(a, p, n)
        loss = loss_fn(a, p, n)
        loss.backward()
        optimizer.step()
        return loss.item()

    return Engine(_update)


def create_triplet_evaluator(model, loss_fn, device=None):
    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            a, p, n = map(lambda x: convert_tensor(x, device), batch)
            a, p, n = model(a, p, n)
            return a, p, n

    engine = Engine(_inference)

    TripletAccuracy().attach(engine, 'accuracy')
    TripletLoss(loss_fn).attach(engine, 'loss')

    return engine


def run(train_batch_size, val_batch_size, epochs, lr, momentum, log_interval, log_dir, weight, prob, args):
    train_loader = get_data_loaders(train_batch_size, prob)
    if weight == '':
        model = FeatureExtractor(
            feature_dim=100) if not args.debug_model else DebugModel()
    else:
        print('loading initial weight from {}'.format(weight))
        loc = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model = torch.load(weight, map_location=loc)
    writer = create_summary_writer(model, train_loader, log_dir)
    device = 'cpu'

    if torch.cuda.is_available():
        device = 'cuda'

    if args.optimizer == 'sgd':
        optimizer = SGD(model.parameters(), lr=lr, momentum=momentum,
                        weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, epochs * len(train_loader.dataset)//train_batch_size)
    else:
        optimizer = Adam(model.parameters(), lr=lr,
                         weight_decay=args.weight_decay)

    loss_fn = TripletLoss(margin=args.margin)
    trainer = create_supervised_trainer(
        model, optimizer, loss_fn, device=device)
    evaluator = create_supervised_evaluator(model,
                                            metrics={'loss': Loss(loss_fn)},
                                            device=device)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1
        if iter % log_interval == 0:
            print("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
                  "".format(engine.state.epoch, iter, len(train_loader), engine.state.output))
            writer.add_scalar(
                "training/loss", engine.state.output, engine.state.iteration)
        if args.optimizer == 'sgd':
            lr_scheduler.step()
            pass

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        if engine.state.epoch % 3 == 0:
            evaluator.run(train_loader)
            metrics = evaluator.state.metrics
            #vg_accuracy = metrics['accuracy']
            avg_loss = metrics['loss']
            # print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
            #       .format(engine.state.epoch, avg_accuracy, avg_loss))
            print("Training Results - Epoch: {}  Avg loss: {:.2f}"
                  .format(engine.state.epoch, avg_loss))
            writer.add_scalar("training/avg_loss",
                              avg_loss, engine.state.epoch)
            # writer.add_scalar("training/avg_accuracy",
            #                   avg_accuracy, engine.state.epoch)
        writer.add_scalar("training/learning_rate",
                          optimizer.param_groups[0]['lr'], engine.state.epoch)
        if args.dataset == 'whale' and engine.state.epoch % 5 == 0:
            train_loader.dataset.sample()
            # loss_fn.increase_difficulty(0.005)
            pass

    def score_function(engine):
        return -evaluator.state.metrics['loss']

    # Setup model checkpoint:
    best_model_saver = ModelCheckpoint(log_dir,
                                       filename_prefix="model",
                                       score_name="train_acc",
                                       score_function=score_function,
                                       n_saved=3,
                                       atomic=True,
                                       create_dir=True,
                                       require_empty=False)
    evaluator.add_event_handler(
        Events.COMPLETED, best_model_saver, {'metric_model': model})

    # @trainer.on(Events.EPOCH_COMPLETED)
    # def log_validation_results(engine):
    #     evaluator.run(val_loader)
    #     metrics = evaluator.state.metrics
    #     avg_accuracy = metrics['accuracy']
    #     avg_nll = metrics['nll']
    #     print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
    #           .format(engine.state.epoch, avg_accuracy, avg_nll))
    #     writer.add_scalar("valdation/avg_loss", avg_nll, engine.state.epoch)
    #     writer.add_scalar("valdation/avg_accuracy",
    #                       avg_accuracy, engine.state.epoch)

    # kick everything off
    trainer.run(train_loader, max_epochs=epochs)

    writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--val-batch-size', type=int, default=1000,
                        help='input batch size for validation (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument("--log-dir", type=str, default="tensorboard_logs",
                        help="log directory for Tensorboard log output")
    parser.add_argument('--weight', type=str, default='',
                        help='initial weight')
    parser.add_argument('--prob', type=float, default=0.1,
                        help='new whaleからデータをサンプリングする確率')
    parser.add_argument('--margin', type=float, default=1.0,
                        help='triplet lossのmarginパラメータ')
    parser.add_argument('--debug-model', action='store_true',
                        help='cpuで挙動確認するようのめちゃ単純なモデルを使う')
    parser.add_argument('--min-size-per-class', type=int, default=1,
                        help='最低min-size-per-class枚以上あるクラスだけを使う')
    parser.add_argument('--weight-decay', type=float, default=0.005,
                        help='weight decay')
    parser.add_argument(
        '--dataset', choices=['whale', 'mnist'], default='whale')
    parser.add_argument('--optimizer', choices=['sgd', 'adam'], default='sgd')

    args = parser.parse_args()
    print(args)

    run(args.batch_size, args.val_batch_size, args.epochs, args.lr, args.momentum,
        args.log_interval, args.log_dir, args.weight, args.prob, args)
