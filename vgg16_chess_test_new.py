import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import image
from matplotlib.ticker import FuncFormatter


loaded_model = load_model('aug_none.h5')

models = []
models.append(load_model('first_vgg16.h5'))
# models.append(load_model('model_lr_0.0005_momentum_0.91.h5'))
# models.append(load_model('model_lr_0.001_momentum_0.9.h5'))
# models.append(load_model('model_lr_0.001_momentum_0.91.h5'))
# models.append(load_model('model_lr_0.002_momentum_0.9.h5'))


img_size = (180, 180)
chess_pieces = ['Bishop', 'King', 'Knight', 'Pawn', 'Queen', 'Rook']
class_accuracies = []
model_accuracies = []


for i in range(len(models)):

    class_accuracies = []

    for chess_piece in chess_pieces:
        directory = f'Chess/{chess_piece}/'
        all_files = os.listdir(directory)

        image_files = [file for file in all_files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
        # image_files = [file for file in all_files if file.lower().endswith(('.png'))]

        correct_predictions = 0
        total_predictions = len(image_files)

        for filename in image_files:
            img_path = os.path.join(directory, filename)
            img = image.load_img(img_path, target_size=img_size)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            predictions = models[i].predict(img_array, verbose = 0)
            predicted_label = np.argmax(predictions)
            
            actual_class_label = chess_piece

            predicted_class_label = chess_pieces[predicted_label]

            if predicted_class_label.lower() == actual_class_label.lower():
                correct_predictions += 1

        accuracy = correct_predictions / total_predictions
        class_accuracies.append(accuracy)
        print(f'Accuracy for {chess_piece}: {accuracy * 100:.2f}%\n')

    overall_accuracy = 0
    for acc in class_accuracies:
        overall_accuracy += acc
    mean_accuracy = (overall_accuracy/6)*100

    model_accuracies.append(class_accuracies)

    print(f"\nMean accuracy for all chess pieces: {mean_accuracy}% \n")

model_accuracies_array = np.array(model_accuracies)
model_accuracies_array = model_accuracies_array.T

print(model_accuracies_array)
num_lists = 5

fig, ax = plt.subplots()
bar_width = 0.35

bars = []
colors = plt.cm.viridis(np.linspace(0, 1, len(chess_pieces)))

for i, piece in enumerate(chess_pieces):
    bars.append(ax.bar(np.arange(num_lists) + bar_width/2, model_accuracies_array[i], bar_width, bottom=np.sum(model_accuracies_array[:i], axis=0), label=piece, color=colors[i]))

# Labels
for bar in ax.patches:
  ax.text(bar.get_x() + bar.get_width() / 2,
          bar.get_height() / 2 + bar.get_y(),
          round(bar.get_height() * 100, 2), ha = 'center',
          color = 'w', weight = 'bold', size = 10)

custom_labels = ['Learning rate = 0.0005\nMomentum = 0.9', 'Learning rate = 0.0005\nMomentum = 0.91', 'Learning rate = 0.001\nMomentum = 0.9',
                'Learning rate = 0.001\nMomentum = 0.91', 'Learning rate = 0.002\nMomentum = 0.9']

ax.set_yticklabels([f'{int(y*100)}' for y in ax.get_yticks()])

ax.set_ylabel('Dokładność klasyfikacji [%]', fontsize = 10, fontweight = 'bold')
ax.set_title('Dokładności dla modeli o różnych parametrach', fontsize = 14, fontweight = 'bold')
ax.set_xticks(np.arange(num_lists) + bar_width / 2)
ax.set_xticklabels(custom_labels, fontsize = 10, fontweight = 'bold')
ax.tick_params(axis='x', which='both', bottom=False)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.subplots_adjust(right=0.8)
print(model_accuracies_array)
plt.show()