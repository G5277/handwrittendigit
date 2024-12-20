def plot_input_img(i,j):
    import matplotlib.pyplot as plt
    plt.imshow(i, cmap = 'binary')
    plt.title(j)
    plt.show()

def test_model(i,j):
    import keras
    model_s = keras.models.load_model("D://Github//handwrittendigit//bestmodel.keras")
    score = model_s.evaluate(i,j)
    return score[1]