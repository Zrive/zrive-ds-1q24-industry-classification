import pandas as pd
import matplotlib.pyplot as plt

def plot_class_data():
    data = {
        'Clase': ['11', '21', '22', '23', '31-33', '42', '44-45', '48-49', '51', '52', '53', '54', '55', '56', '61', '62', '71', '72', '81', '92'],
        'F1-Score': [0.5692, 0.6471, 0.5498, 0.7663, 0.6106, 0.357, 0.2332, 0.431655, 0.286, 0.507, 0.416, 0.7627, 0.7636, 0.48, 0.5253, 0.4725, 0.1946, 0.5393, 0.188, 0.7912],
        'AUC': [0.6035, 0.7317, 0.6239, 0.7948, 0.68, 0.38, 0.3, 0.63, 0.3, 0.62, 0.6051, 0.83, 0.7735, 0.52, 0.5649, 0.5831, 0.23, 0.6008, 0.24, 0.8216],
        'Descripciones Únicas': [368, 257, 276, 3830, 2180, 802, 497, 721, 404, 268, 482, 5547, 275, 1367, 802, 592, 304, 373, 539, 230],
        'Número de Subclases': [120, 21, 14, 31, 270, 72, 104, 85, 30, 35, 32, 89, 3, 40, 19, 48, 23, 17, 41, 32]
    }

    df = pd.DataFrame(data)
    df = df.sort_values(by='F1-Score', ascending=False)

    fig, ax1 = plt.subplots(figsize=(12, 8))

    indices = range(len(df))
    ax1.plot(indices, df['F1-Score'], 'bo-', label='F1-Score', linewidth=2, markersize=8)
    ax1.plot(indices, df['AUC'], 'go-', label='AUC', linewidth=2, markersize=8)
    ax1.set_xlabel('Clase')
    ax1.set_ylabel('Ratio')
    ax1.set_title('Model Performance and Class Complexity')
    ax1.set_xticks(indices)
    ax1.set_xticklabels(df['Clase'])
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(indices, df['Descripciones Únicas'], 'ks-', label='Descripciones Únicas', linewidth=2, markersize=8)
    ax2.set_ylabel('Descripciones Únicas')

    ax3 = ax1.twinx()
    ax3.plot(indices, df['Número de Subclases'], 'ro-', label='Número de Subclases', linewidth=2, markersize=8)
    ax3.set_ylabel('Número de Subclases', color='red')
    ax3.spines['right'].set_position(('outward', 60))
    ax3.set_ylim(0, 300)
    ax3.set_yticks(range(0, 301, 25))
    ax3.tick_params(axis='y', colors='red')

    ax3.legend(handles=[ax1.lines[0], ax1.lines[1], ax2.lines[0], ax3.lines[0]], loc='upper right')
    plt.savefig('conclutions_noise_graph.png')
    plt.show()

