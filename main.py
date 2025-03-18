from attacks import bim, utils
import config

if __name__ == "__main__":

    img_loader = utils.Preprocessor(config.IMG_PATH)

    train_data, test_data = img_loader.load_images()
    model = img_loader.get_model()

    iterative_attack = bim.BIM(epsilon= .5, input_label=103, model = model, steps=30)

    iterative_attack.generate(train_data, targeted=True)