# ELECTRA 모델링
huggingface 라이브러리를 활용한 [KoELECTRA](https://huggingface.co/monologg/koelectra-base-v3-discriminator) 모델링

## Requirements
- python==3.9.12
- transformers==4.14.1
- pytorch==1.10.1
- cudatoolkit==11.3.1

### `conda`를 활용한 환경 구성
```commandline
$ conda create --name <env> --file <this file>
```

## Usage
### Modeling
```commandline
$ python text_classification.py
```
#### Output 구성
```
.
└── checkpoint-500
    ├── config.json  # 추론에 필수
    ├── optimizer.pt
    ├── pytorch_model.bin  # 추론에 필수
    ├── rng_state.pth
    ├── scheduler.pt
    ├── special_tokens_map.json
    ├── tokenizer_config.json  # 추론에 필수
    ├── tokenizer.json  # 추론에 필수
    ├── trainer_state.json
    ├── training_args.bin
    └── vocab.txt
```
### Inference
서버용, TorchScript, 경량화 3개 버전 추론
```commandline
$ python inference.py
```
### Embedded
`TorchScript`, `Post Training Dynamic Quantization`를 활용한 임베디드 및 경량화
```commandline
$ python optimize.py
```
### Dockerize
`./app` 폴더에서 배포하고자 하는 환경에 맞춰 `main_server.py` 혹은 `main_edge.py` 파일명을 `main.py`로 변경한 후 도커 빌드


## Reference
- [Fine-tune](https://huggingface.co/docs/transformers/training)
- [Text classification](https://huggingface.co/docs/transformers/tasks/sequence_classification)
- [Pipelines for inference](https://huggingface.co/docs/transformers/pipeline_tutorial#pipeline-usage)
- [Tokenizer](https://huggingface.co/docs/transformers/preprocessing#nlp)
- [TorchScript](https://huggingface.co/docs/transformers/serialization#torchscript)
- [Qunatization](https://pytorch.org/tutorials/recipes/script_optimized.html#optimize-a-torchscript-model)
- [How to resume training](https://github.com/huggingface/transformers/issues/7198#issuecomment-694352941)
- [Token classification metrics](https://huggingface.co/course/chapter7/2#metrics)
