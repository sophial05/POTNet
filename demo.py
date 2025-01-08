from potnet import POTNet
from examples.utils import generate_ground_truth
from matplotlib import pyplot as plt


if __name__ == '__main__':
    gt_Xs, gt_thetas = generate_ground_truth(3000)
    potnet_model = POTNet(embedding_dim = gt_thetas.shape[1],
                          epochs = 100,
                          verbose=True)

    potnet_model.fit(gt_thetas, 
                    epochs=100,
                    save_checkpoint=True,
                    checkpoint_epoch=50,
                    overwrite_checkpoint=False)
    potnet_gen_data = potnet_model.generate(3000)
    potnet_model.save(model_path = 'potnet.pkl')