import argparse
from pytorch_lightning import Trainer
from dataloader import SepsisDataloader 
from BDLSTM import BDLSTM
from CNN_LSTM import BDLSTM2
import multiprocessing
#import wandb
import torch

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

    # Check if a GPU is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare data
    data_module = SepsisDataloader(data_dir, window_size, batch_size, num_workers, device=device)
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # Instantiate BDLSTMs
    model = BDLSTM(input_size, hidden_size, output_size, lr=lr, stateless=stateless)
    model_1 = BDLSTM2(input_size=input_size, lookback=10, filters1=64, filters2=64,
                      filters3=64, hidden_size=hidden_size, dropout=0.3, lr=lr)

    # Initialize Trainer
    trainer = Trainer(max_epochs=num_epochs)

    # Start training
    trainer.fit(model, train_loader, val_loader)
    trainer.fit(model_1, train_loader, val_loader)

    FILE = "Sepsis_BDLSTM_model.pth"
    FILE_1 = "Sepsis_BDLSTM2_model.pth"
    torch.save(model, FILE)
    torch.save(model_1, FILE_1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sepsis Model Training')

    parser.add_argument('--window_size', type=int, default=13, help='Rolling w  indow size')
    parser.add_argument('--data_dir', type=str, default='C:\\Users\\JanFrackowiak\\Desktop\\Miscelleneous\\Sepsis-prediction-with-NNs\\training', help='Directory with data')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loader workers')
    parser.add_argument('--input_size', type=int, default=40, help='Input size for the model')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size for the model')
    parser.add_argument('--output_size', type=int, default=1, help='Output size for the model')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--stateless', type=bool, default=True, help='Whether the model is stateless')
    parser.add_argument('--num_epochs', type=int, default=12, help='Number of epochs')
    
    args = parser.parse_args()
    
    main(args)

