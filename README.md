# Defending Your Voice: Adversarial Attack on Voice Conversion

This is the official implementation of the paper [Defending Your Voice: Adversarial Attack on Voice Conversion](https://arxiv.org/abs/2005.08781).
We use adversarial attack to prevent one's voice from improperly utilized in voice conversion.
The conversion fails when adversarial noise is added to the input utterance.

For the audio samples, please refer to our [demo page](https://yistlin.github.io/attack-vc-demo/).

## Pre-trained Model

The pre-trained [AdaIN-VC](https://arxiv.org/abs/1904.05742) (referred to as Chou's model in the paper) is available [here](https://drive.google.com/file/d/1vBF-4s5u0sro3nwDFWL7VnAV6KViCMp0/view?usp=sharing).
The files are from the official implementation of AdaIN-VC, but the config file was slightly modified.

## Attack

You can perform adversarial attack on AdaIN-VC with `attack.py`.

```bash
python attack.py <model_dir> <vc_tgt> <adv_tgt> <output> [--vc_src source] [--eps epsilon] [--n_iters iterations] [--attack_type type]
```

- **model_dir**: The directory of model files.
- **vc_tgt**: The target utterance to be defended, providing vocal timbre in voice conversion.
- **adv_tgt**: The target used in adversarial attack (**y** in the paper).
- **output**: The output defended utterance.
- **source**: The source utterance providing linguistic content in voice conversion (required in end-to-end and feedback attack).
- **epsilon**: The maximum amplitude of the perturbation.
- **iterations**: The number of iterations for updating the perturbation.
- **type**: The type of adversarial attack to use (end-to-end, embedding, or feedback attack).

## Inference

You can perform voice conversion with `inference.py`.

```bash
python inference.py <model_dir> <source> <target> <output>
```

- **model_dir**: The directory of model files.
- **source**: The source utterance providing linguistic content in voice conversion.
- **target**: The target utterance providing vocal timbre in voice conversion.
- **output**: The output converted utterance.

## Reference

Please cite our paper if you find it useful.

```bib
@INPROCEEDINGS{9383529,
  author={C. -y. {Huang} and Y. Y. {Lin} and H. -y. {Lee} and L. -s. {Lee}},
  booktitle={2021 IEEE Spoken Language Technology Workshop (SLT)},
  title={Defending Your Voice: Adversarial Attack on Voice Conversion},
  year={2021},
  volume={},
  number={},
  pages={552-559},
  doi={10.1109/SLT48900.2021.9383529}}
```

##Sections annexed by mohadese shabani 

1. A summary of the purpose as well as the function of the code

In this project we are trying to report the hostile attack on audio conversion. At first, subtle human disturbances are added to the sentences that the speaker defends, then with three different approaches, end-to-end attack, embedded sentence and feedback attack, which are proposed in this project, so that the characteristics of the speaker of the sentences become very different from the speaker. It is reported that they have been defended.Regarding the function of the code, we can say that using the attack.py file codes, we apply a hostile attack on the sound, and then use the inference.py file codes to perform audio conversion.


2. Explanation of innovation in code improvement





