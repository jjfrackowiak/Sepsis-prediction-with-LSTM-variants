import argparse
from pytorch_lightning import Trainer
from dataloader import SepsisDataloader 
from BDLSTM import BDLSTM
import multiprocessing
import wandb

def main(args):
    data_dir = args.data_dir
    window_size = args.window_size
    batch_size = args.batch_size
    num_workers = args.num_workers
    input_size = args.input_size
    hidden_size = args.hidden_size
    output_size = args.output_size
    lr = args.lr
    stateless = args.stateless
    num_epochs = args.num_epochs

    # Prepare data
    data_module = SepsisDataloader(data_dir, window_size, batch_size, num_workers)
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    print(next(iter(train_loader)))

    # Instantiate model
    model = BDLSTM(input_size, hidden_size, output_size, lr=1e-3, stateless=True)

    # Initialize Trainer
    trainer = Trainer(max_epochs=num_epochs)
    #trainer = Trainer(max_epochs=num_epochs, gpus=1 if device.type == "cuda" else 0)  # Set gpus to 1 if using GPU, 0 otherwise


    # Start training
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sepsis Model Training')

    parser.add_argument('--window_size', type=int, default=13, help='Window size')
    parser.add_argument('--data_dir', type=str, default='../../files/challenge-2019/1.0.0/training', help='Directory with data')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loader workers')
    parser.add_argument('--input_size', type=int, default=40, help='Input size for the model')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size for the model')
    parser.add_argument('--output_size', type=int, default=1, help='Output size for the model')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--stateless', type=bool, default=True, help='Whether the model is stateless')
    parser.add_argument('--num_epochs', type=int, default=12, help='Number of epochs')
    
    args = parser.parse_args()
    
    main(args)

