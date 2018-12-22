from PIL import Image
from argparse import ArgumentParser
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import SGD
from torchvision import transforms
from tensorboardX import SummaryWriter
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import ModelCheckpoint

from data_loader import LabeledImageDataset
from model.gap_resnet import GapResnet


def get_data_loaders(train_batch_size):

    train_data_transform = transforms.Compose([
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20, resample=Image.BILINEAR),
        transforms.Resize((384, 384)),
        #        transforms.RandomCrop((200, 800), pad_if_needed=True),
        transforms.ToTensor()
    ])

    train_loader = DataLoader(LabeledImageDataset('data', transform=train_data_transform),
                              batch_size=train_batch_size, shuffle=True)
    return train_loader


def create_summary_writer(model, data_loader, log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    data_loader_iter = iter(data_loader)
    x, y = next(data_loader_iter)
    try:
        writer.add_graph(model, x)
    except Exception as e:
        print("Failed to save model graph: {}".format(e))
    return writer


def run(train_batch_size, val_batch_size, epochs, lr, momentum, log_interval, log_dir, weight):
    train_loader = get_data_loaders(train_batch_size)
    if weight == '':
        model = GapResnet()
    else:
        print('loading initial weight from {}'.format(weight))
        loc = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model = torch.load(weight, map_location=loc)
    writer = create_summary_writer(model, train_loader, log_dir)
    device = 'cpu'

    if torch.cuda.is_available():
        device = 'cuda'

    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epochs * len(train_loader.dataset)//train_batch_size)
    trainer = create_supervised_trainer(
        model, optimizer, F.cross_entropy, device=device)
    evaluator = create_supervised_evaluator(model,
                                            metrics={'accuracy': Accuracy(),
                                                     'nll': Loss(F.cross_entropy)},
                                            device=device)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1
        if iter % log_interval == 0:
            print("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
                  "".format(engine.state.epoch, iter, len(train_loader), engine.state.output))
            writer.add_scalar(
                "training/loss", engine.state.output, engine.state.iteration)
        lr_scheduler.step()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        if engine.state.epoch % 5 == 0:
            evaluator.run(train_loader)
            metrics = evaluator.state.metrics
            avg_accuracy = metrics['accuracy']
            avg_nll = metrics['nll']
            print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
                  .format(engine.state.epoch, avg_accuracy, avg_nll))
            writer.add_scalar("training/avg_loss", avg_nll, engine.state.epoch)
            writer.add_scalar("training/avg_accuracy",
                              avg_accuracy, engine.state.epoch)
        writer.add_scalar("training/learning_rate",
                          optimizer.param_groups[0]['lr'], engine.state.epoch)
        train_loader.dataset.rotate_new_whale()

    def score_function(engine):
        return evaluator.state.metrics['accuracy']

    # Setup model checkpoint:
    best_model_saver = ModelCheckpoint(log_dir,
                                       filename_prefix="model",
                                       score_name="train_acc",
                                       score_function=score_function,
                                       n_saved=3,
                                       atomic=True,
                                       create_dir=True)
    evaluator.add_event_handler(
        Events.COMPLETED, best_model_saver, {'gap_resnet': model})

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
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--val_batch_size', type=int, default=1000,
                        help='input batch size for validation (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument("--log_dir", type=str, default="tensorboard_logs",
                        help="log directory for Tensorboard log output")
    parser.add_argument('--weight', type=str, default='',
                        help='initial weight')

    args = parser.parse_args()

    run(args.batch_size, args.val_batch_size, args.epochs, args.lr, args.momentum,
        args.log_interval, args.log_dir, args.weight)
