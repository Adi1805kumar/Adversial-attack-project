# =====================================================================================================
# PASTE MATERIAL - not meant to be run directly. Two parts: one for configs.py, one for experiments.py.
# Also copy stickers.py next to attacks.py.
# =====================================================================================================


# =====================================================================================================
# PART A - append at the bottom of configs.py
# =====================================================================================================

# Universal sticker attack. sizes are (height, width) in pixels; region = (top, left, bottom, right) is the area
# where stickers may be placed (the "merged mask" of the paper - for GTSRB crops the centre lies on the sign).
sticker_normalization = {
    "MNIST": ((0.0,), (1.0,)),  # MNIST is not normalized
    "traffic_signs": ((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669)),  # same numbers as data_transform
}
sticker_hps = {
    "MNIST": {"sizes": [(4, 4), (6, 6), (8, 8), (10, 10)], "region": (4, 4, 24, 24), "stride": 2},
    "traffic_signs": {"sizes": [(6, 6), (8, 8), (10, 10), (12, 12)], "region": (6, 6, 26, 26), "stride": 2},
}
sticker_color_variants = [["black"], ["white"], ["black", "white"]]  # one entry = one sticker
sticker_search_images = 500  # correctly classified images used to search the sticker location


# =====================================================================================================
# PART B - changes to experiments.py
# =====================================================================================================

# B1. add to the imports at the top:
#       import stickers
#       from torch.utils.data import DataLoader

# B2. add next to the other flags (run_experiment_1 / 2 / 3):
#       run_experiment_sticker = True

# B3. append at the very END of the file:

def experiment_sticker_func(net_names, _training_dataset, _testing_dataset, save_plots=False, show_plots=False):
    """
    Universal sticker attack on already trained networks (loaded from the checkpoints saved by experiments 1 and 2).
    For every network and sticker variant: search the sticker location on training images, then measure the
    accuracy under the sticker on the held-out test set - the same quantity reported for FGSM and PGD.
    """
    mean, std = configs.sticker_normalization[dataset_name]
    test_dl = DataLoader(_testing_dataset, batch_size=100)
    results = {}
    for net_name in net_names:
        checkpoint_path = os.path.join(experiment_checkpoints_folder, "{}.pt".format(net_name))
        if not os.path.exists(checkpoint_path):
            logger.log_print("skip {}: no checkpoint at {} (run experiments 1 and 2 with save_checkpoints=True)"
                             .format(net_name, checkpoint_path))
            continue

        net = network_architecture().to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        net.load_state_dict(checkpoint["trained_net"])
        net.eval()

        clean_acc = trainer.measure_classification_accuracy(net, test_dl, device=device)
        logger.log_print("\nSticker attack on {} (clean test accuracy: {})".format(net_name, clean_acc))
        _, search_dl = datasets.get_train_val_dls(_training_dataset, checkpoint["net_hp"]["batch_size"])

        for colors in configs.sticker_color_variants:
            hp = dict(configs.sticker_hps[dataset_name])
            hp["colors"] = colors
            attack = stickers.UniversalSticker(net, _loss_fn, hp, mean=mean, std=std)
            attack.fit(search_dl, max_images=configs.sticker_search_images, device=device)

            title = "{}_{}_{}".format(attack.name, "+".join(colors), net_name)
            acc_under_sticker = attack.test_attack(test_dl, main_title=title, plot_results=show_plots,
                                                   save_results_figs=save_plots,
                                                   fig_path=os.path.join(plots_folder, "{}.png".format(title)),
                                                   device=device)
            top_wrong_labels = attack.wrong_label_counts(test_dl, device=device)[:3]
            logger.log_print("  stickers {} at {}".format(colors, attack.placements))
            logger.log_print("  accuracy under sticker: {}   (most common wrong labels: {})"
                             .format(acc_under_sticker, top_wrong_labels))
            results[(net_name, "+".join(colors))] = acc_under_sticker

    return results


# Experiment 5: universal sticker attack (black-box, input-agnostic) on the standard and adversarially trained nets
if __name__ == '__main__' and run_experiment_sticker:
    logger.new_section()
    logger.log_print("Experiment 5: universal sticker attack")
    base_name = network_architecture.name
    experiment_sticker_func([base_name,
                             "{} with FGSM adversarial training".format(base_name),
                             "{} with PGD adversarial training".format(base_name)],
                            _training_dataset, _testing_dataset,
                            save_plots=configs.save_attacks_plots, show_plots=configs.show_attacks_plots)
