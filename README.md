# TeluguASR

This is a Repository for Telugu Autotmatic Speech recognition (ASR) model with `facebook/wav2vec2-large-xlsr-53` trained on IIIT Hyderabad ASR Challenge dataset and OpenSLR66 Telugu dataset. When using this model, make sure that your speech input is sampled at 16kHz.

Trained model can be found at `https://huggingface.co/henilp105/wav2vec2-large-xls-r-300m-telugu-asr`.

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0003
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 2
- total_train_batch_size: 16
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 500
- num_epochs: 150
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss | Wer    |
|:-------------:|:-----:|:----:|:---------------:|:------:|
| 6.0506        | 2.3   | 200  | 0.8841          | 0.7564 |
| 0.6354        | 4.59  | 400  | 0.7448          | 0.6912 |
| 0.3934        | 6.89  | 600  | 0.8321          | 0.6929 |
| 0.2652        | 9.19  | 800  | 0.9529          | 0.6984 |
| 0.2022        | 11.49 | 1000 | 0.9490          | 0.6979 |
| 0.1514        | 13.79 | 1200 | 1.0025          | 0.6869 |
| 0.124         | 16.09 | 1400 | 1.0367          | 0.6799 |
| 0.1007        | 18.39 | 1600 | 1.0658          | 0.6734 |
| 0.0875        | 20.69 | 1800 | 1.0758          | 0.6779 |
| 0.0838        | 22.98 | 2000 | 1.0999          | 0.6701 |
| 0.0745        | 25.29 | 2200 | 1.1020          | 0.6708 |
| 0.0641        | 27.58 | 2400 | 1.1140          | 0.6683 |
| 0.0607        | 29.88 | 2600 | 1.1050          | 0.6656 |
| 0.0607        | 29.88 | 2600 | 1.1050          | 0.6656 |


## Usage
The model can be used directly (without a language model) as follows:
```python
import torch
import torchaudio
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import pandas as pd

# Evaluation notebook contains the procedure to download the data
df = pd.read_csv("/content/te/test.tsv", sep="\t")
df["path"] = "/content/te/clips/" + df["path"]
test_dataset = Dataset.from_pandas(df)
processor = Wav2Vec2Processor.from_pretrained("henilp105/wav2vec2-large-xlsr-53-telugu")
model = Wav2Vec2ForCTC.from_pretrained("henilp105/wav2vec2-large-xlsr-53-telugu") 
resampler = torchaudio.transforms.Resample(48_000, 16_000)

# Preprocessing the datasets.
def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = resampler(speech_array).squeeze().numpy()
    return batch
test_dataset = test_dataset.map(speech_file_to_array_fn)
inputs = processor(test_dataset["speech"][:2], sampling_rate=16_000, return_tensors="pt", padding=True)
with torch.no_grad():
    logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
predicted_ids = torch.argmax(logits, dim=-1)
print("Prediction:", processor.batch_decode(predicted_ids))
print("Reference:", test_dataset["sentence"][:2])
```

## Evaluation
```python
import torch
import torchaudio
from datasets import Dataset, load_metric
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import re
from sklearn.model_selection import train_test_split
import pandas as pd

# Evaluation notebook contains the procedure to download the data
df = pd.read_csv("/content/te/test.tsv", sep="\t")
df["path"] = "/content/te/clips/" + df["path"]
test_dataset = Dataset.from_pandas(df)
wer = load_metric("wer")
processor = Wav2Vec2Processor.from_pretrained("henilp105/wav2vec2-large-xlsr-53-telugu")
model = Wav2Vec2ForCTC.from_pretrained("henilp105/wav2vec2-large-xlsr-53-telugu") 
model.to("cuda")
chars_to_ignore_regex = '[\,\?\.\!\-\_\;\:\"\“\%\‘\”\।\’\'\&]'
resampler = torchaudio.transforms.Resample(48_000, 16_000)
def normalizer(text):
    text = text.replace("\\n","\n")
    text = ' '.join(text.split())
    text = re.sub(r'''([a-z]+)''','',text,flags=re.IGNORECASE)
    text = re.sub(r'''%'''," శాతం ", text)
    text = re.sub(r'''(/|-|_)'''," ", text)
    text = re.sub("ై","ై", text)
    text = text.strip()
    return text

def speech_file_to_array_fn(batch):
    batch["sentence"] = normalizer(batch["sentence"])
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()+ " "
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = resampler(speech_array).squeeze().numpy()
    return batch
test_dataset = test_dataset.map(speech_file_to_array_fn)

# Preprocessing the datasets.
def evaluate(batch):
    inputs = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values.to("cuda"), attention_mask=inputs.attention_mask.to("cuda")).logits
    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_strings"] = processor.batch_decode(pred_ids)
    return batch
result = test_dataset.map(evaluate, batched=True, batch_size=8)
print("WER: {:2f}".format(100 * wer.compute(predictions=result["pred_strings"], references=result["sentence"])))
```

**Test Result WER**: 33.44%

## Training

70% of the OpenSLR Telugu  and 40% of IIITH ASR Challenge dataset was used for training.

Training Data Preparation notebook can be found [here](prepare_dataset.ipynb)

Training notebook can be found [here](training.ipynb)

Evaluation notebook is [here](evaluation.ipynb)

Evalution is purely done on the rest 60% of the IIITH ASR Challenge dataset.

### Framework versions

- Transformers 4.24.0
- Pytorch 1.10.0+cu113
- Datasets 1.18.3
- Tokenizers 0.13.2