import torch
import torch.optim as optim
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from data_utils import get_data_loaders
from models.model import SimpleNN
from utils import setup_logging, log_decorator, load_config, save_checkpoint

config = load_config()

if config['tensorboard_logging']:
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir='logs/tensorboard')


@log_decorator
def train_model(model, train_loader, val_loader, num_epochs, lr, early_stopping, lr_scheduler, save_checkpoint_freq):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    if lr_scheduler:
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop_patience = 10  # Number of epochs to wait for improvement before stopping

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}')

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')

        if config['tensorboard_logging']:
            writer.add_scalar('Loss/train', avg_train_loss, epoch)
            writer.add_scalar('Loss/val', avg_val_loss, epoch)

        # Check for early stopping
        if early_stopping:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stop_patience:
                    logging.info(f'Early stopping at epoch {epoch + 1}')
                    break

        # Learning rate scheduling
        if lr_scheduler:
            scheduler.step()

        # Save checkpoint
        if save_checkpoint_freq > 0 and (epoch + 1) % save_checkpoint_freq == 0:
            save_checkpoint(model, epoch + 1)

    plot_loss(train_losses, val_losses)
    return model


def plot_loss(train_losses, val_losses):
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()


@log_decorator
def main():
    setup_logging(config['log_filename'])

    train_loader, val_loader, test_loader = get_data_loaders()

    # Check data sizes and visualize
    inputs, labels = next(iter(train_loader))
    print(f"Input batch shape: {inputs.shape}")
    print(f"Labels batch shape: {labels.shape}")

    # Visualize some data samples
    plt.figure(figsize=(10, 4))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(inputs[i].numpy().squeeze(), cmap='gray')
        plt.title(f'Label: {labels[i].item()}')
        plt.axis('off')
    plt.show()

    # Initialize and test model
    model = SimpleNN()
    test_input = torch.randn(1, 1, 28, 28)
    test_output = model(test_input)
    print(f"Test output shape: {test_output.shape}")

    # Train model
    trained_model = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=config['num_epochs'],
        lr=config['learning_rate'],
        early_stopping=config['early_stopping'],
        lr_scheduler=config['lr_scheduler'],
        save_checkpoint_freq=config['save_checkpoint']
    )

    # Save final model
    torch.save(trained_model.state_dict(), 'model.pth')
    logging.info('Model saved to model.pth')


if __name__ == '__main__':
    main()
