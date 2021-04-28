import matplotlib.pyplot as plt


def create_history_plot(history, model_name, metrics=None):
    plt.title('Accuracy and Loss ({})'.format(model_name))
    if metrics == None:
        metrics = {'accuracy', 'loss'}
    
    if 'acc' in metrics:
        plt.plot(history.history['accuracy'], color='g', label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], color='b', label='Validation Accuracy')

    if 'loss' in metrics:
        plt.plot(history.history['loss'], color='r', label='Train Loss')
        plt.plot(history.history['val_loss'], color='m', label='Validation Loss')

    plt.legend(loc='best')
    plt.tight_layout()


def plot_history(history, model_name):
    create_history_plot(history, model_name)
    plt.show()


def plot_and_save_history(history, model_name, file_path, metrics=None):
    if metrics == None:
        metrics = {'accuracy', 'loss'}
    create_history_plot(history, model_name, metrics)
    plt.savefig(file_path)