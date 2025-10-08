# Phi-4-Ads

## Background

This directory contains everything needed to run the lightweight fine-tuned model on conversation rounds from our chatbot advertising user study.

## Installation

`conda create -n phi4ads python=3.10`
`conda activate phi4ads`
`pip install -r requirements.txt`

Link to weights and data: [https://drive.google.com/drive/folders/1a4xbkKwJ4UnzfFE6RuQVWBAkZezj7edQ?usp=drive_link](https://drive.google.com/drive/folders/1a4xbkKwJ4UnzfFE6RuQVWBAkZezj7edQ?usp=drive_link)

Unzip the weights (.tar.gz) into the `phi-4-ads/` directory. There should be a folder containing the weights `Phi-4-Ads-8192`.

## Basic Usage

### Inference

With the unzipped weights, you can run `inference_chatbot.py` to get a command line interface with the Phi-4-ads model. You can run it with the `--product "product name"` and `--ads-enabled` flags for the specific product to advertise in a session. The subtlety and relevance of the ad will depend on the topic/task, which has not yet been thoroughly evaluated.

### Fine-Tuning

Download the data `ranked_generations.json` into the `phi-4-ads/` directory. Running `finetune_phi4.py` should download the base Phi-4 weights and architecture, as well as the orca-agentinstruct-1M-v1-cleaned dataset (which we only sample 250 conversations of during training). The model is not trained to produce reasoning tokens.

## Licensing

This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.

The GNU General Public License (GPL) is a copyleft license that allows users to distribute and modify the software as long as they make their modifications available under the same terms. This means that any derivative work must also be licensed under the GPL.

For more information on the GPL license, please visit https://www.gnu.org/licenses/gpl-3.0.en.html.
