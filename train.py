import os

import pandas as pd
import torch

from src.models import ResNet
from src.train_utils import DataHandler, Trainer

MODEL_WEIGHTS_DIR = 'model_weights'
CHECKPOINT_DIR = 'checkpoints'
TRAIN_FILE = 'data/processed/lichess_Chess-Network_2022.csv'
TARGET_PLAYER = 'Chess-Network'
HISTORY_SIZE = 12
MIN_CLOCK_SECONDS = 15.
N_GAMES_BATCH = 10

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.00001
BATCH_SIZE = 1024
ACCUMULATION_STEPS = 2
N_EPOCHS = 100
CHECKPOINT_EPOCHS = 10

if __name__ == "__main__":
    df_states_train = pd.read_csv(TRAIN_FILE)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet.init_standard(history_size=HISTORY_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    model_file_name = f'resnet_{TARGET_PLAYER}_{HISTORY_SIZE}.pt'
    checkpoint_path = f"./{CHECKPOINT_DIR}/{model_file_name}"
    starting_epoch = 0
    if model_file_name in os.listdir(CHECKPOINT_DIR):
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch']
        print(f'Model and optimizer initialized from {checkpoint_path}')
        del checkpoint

    data_handler = DataHandler(df_states=df_states_train, history_size=HISTORY_SIZE)
    trainer = Trainer(batch_size=BATCH_SIZE, accumulation_steps=ACCUMULATION_STEPS)

    for epoch in range(starting_epoch, starting_epoch + N_EPOCHS):
        train_data = data_handler.get_encoded_games_batch(n_games=N_GAMES_BATCH, target_player=TARGET_PLAYER,
                                                          min_clock_seconds=MIN_CLOCK_SECONDS)
        loss = trainer.train_iteration(model, optimizer, train_data, device)
        print("Epoch = %5d, train loss = %5.3e" % (epoch + 1, loss))
        del train_data

        if (epoch + 1) % CHECKPOINT_EPOCHS == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            model_path = f"./{MODEL_WEIGHTS_DIR}/{model_file_name}"
            torch.save(model.state_dict(), model_path)
