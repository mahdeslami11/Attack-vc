import argparse

import soundfile as sf
import torch

from attack_utils import e2e_attack, emb_attack, fb_attack
from data_utils import denormalize, file2mel, load_model, mel2wav, normalize


def main(
    model_dir: str,
    vc_src: str,
    vc_tgt: str,
    adv_tgt: str,
    output: str,
    eps: float,
    n_iters: int,
    attack_type: str,
):
    assert attack_type == "emb" or vc_src is not None
    model, config, attr, device = load_model(model_dir)

    vc_tgt = file2mel(vc_tgt, **config["preprocess"])
    adv_tgt = file2mel(adv_tgt, **config["preprocess"])

    vc_tgt = normalize(vc_tgt, attr)
    adv_tgt = normalize(adv_tgt, attr)

    vc_tgt = torch.from_numpy(vc_tgt).T.unsqueeze(0).to(device)
    adv_tgt = torch.from_numpy(adv_tgt).T.unsqueeze(0).to(device)

    if attack_type != "emb":
        vc_src = file2mel(vc_src, **config["preprocess"])
        vc_src = normalize(vc_src, attr)
        vc_src = torch.from_numpy(vc_src).T.unsqueeze(0).to(device)

    if attack_type == "e2e":
        adv_inp = e2e_attack(model, vc_src, vc_tgt, adv_tgt, eps, n_iters)
    elif attack_type == "emb":
        adv_inp = emb_attack(model, vc_tgt, adv_tgt, eps, n_iters)
    elif attack_type == "fb":
        adv_inp = fb_attack(model, vc_src, vc_tgt, adv_tgt, eps, n_iters)
    else:
        raise NotImplementedError()

    adv_inp = adv_inp.squeeze(0).T
    adv_inp = denormalize(adv_inp.data.cpu().numpy(), attr)
    adv_inp = mel2wav(adv_inp, **config["preprocess"])

    sf.write(output, adv_inp, config["preprocess"]["sample_rate"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=str, help="The directory of model files.")
    parser.add_argument(
        "vc_tgt",
        type=str,
        help="The target utterance to be defended, providing vocal timbre in voice conversion.",
    )
    parser.add_argument(
        "adv_tgt", type=str, help="The target used in adversarial attack."
    )
    parser.add_argument("output", type=str, help="The output defended utterance.")
    parser.add_argument(
        "--vc_src",
        type=str,
        default=None,
        help="The source utterance providing linguistic content in voice conversion (required in end-to-end and feedback attack).",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.1,
        help="The maximum amplitude of the perturbation.",
    )
    parser.add_argument(
        "--n_iters",
        type=int,
        default=1500,
        help="The number of iterations for updating the perturbation.",
    )
    parser.add_argument(
        "--attack_type",
        type=str,
        choices=["e2e", "emb", "fb"],
        default="emb",
        help="The type of adversarial attack to use (end-to-end, embedding, or feedback attack).",
    )
    main(**vars(parser.parse_args()))
