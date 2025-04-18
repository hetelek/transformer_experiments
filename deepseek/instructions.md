I'm trying to run the largest DeepSeek model on my Macbook M1 Pro.I understand it'll be slow. My approach is to page swap the weights from disk layer-by-layer to do a forward pass.

It should have at most, say, 10GB of weights in RAM at a time. We need to modify deepseek.py. Somehow modify it to load 1 layer at a time into RAM and the continue.

For testing, I created a `config_671B_custom.json` which sets `n_layer` to 4 which is wrong, and removes the quantization `("dtype": "fp8")` because the MPS MacOS can't do it, so we just disable for now. You will likely need to update the json file to the original n_layers (check `config_671B.json`) but probably keep `"dtype": "fp8"` removed.

Can you try to do this? Run it and iterate as needed.
