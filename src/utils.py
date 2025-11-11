import os
import matplotlib.pyplot as plt

def save_plots(args, train_loss, train_accuracy, val_loss, val_accuracy):
    os.makedirs(args.output_dir, exist_ok=True)

    plt.figure()
    plt.plot(train_loss, label='Train Loss', color='Green')
    plt.plot(val_loss, label='Validation Loss', color='Blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'loss_curve.png'))
    plt.close()

    plt.figure()
    plt.plot(train_accuracy, label='Train Accuracy', color='Green')
    plt.plot(val_accuracy, label='Validation Accuracy', color='Blue')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'accuracy_curve.png'))