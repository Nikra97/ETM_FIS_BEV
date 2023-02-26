# Dataset Preparation

## Dataset structure

It is recommended to symlink the dataset root to `$BEVerse/data`.
If your folder structure is different from the following, you may need to change the corresponding paths in config files.

```
BEVerse
├── mmdet3d
├── tools
├── configs
├── projects
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
```

## Download and prepare the nuScenes dataset

Download nuScenes V1.0 full dataset data [HERE](https://www.nuscenes.org/download), including the map extensions. Prepare nuscenes data by running

```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```


wget -O "v1.0-trainval_meta.tgz" "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval_meta.tgz?AWSAccessKeyId=ASIA6RIK4RRMEER7HDIM&Signature=N9U2GOHlw4vLtNkk00xwDuAm85Q%3D&x-amz-security-token=IQoJb3JpZ2luX2VjECsaCXVzLWVhc3QtMSJHMEUCIQCUSsn4qisaV18V1UO2tiDVSCA1VZyqSOkjdC03CNKQvAIgOPrs3hNMDwYp0lcVsdFHDAguLUVlYroQl2zNQh4XKVQq9AIIMxADGgw5OTkxMzk2MDk2ODgiDDy%2BVsEHwFLMc45ToyrRAqatoWIyg2Mi35HkDgIHNxUal16zPRD3tGyXwJSWqCWIsW%2BC9miqHEPqsJPVvlSjhM4j7J0SLMGwFjnJuOWeOyZwoN9D4FJ6lv0qdhOCpBp%2FkP9E0xwKME0RgftxK478Hmj%2FHMyJHW%2B4kqHd6%2FimJxZvNjF7ZSdA0KNyD7V%2BIrbdoQdufSTonEZqqUtVL0sQgAb3iGEUOm6xd2Re8P0krJC8YHhko7KKOf%2FdY%2FmgslwjR2%2FaSaqB3F%2B5pP5WX3DEvJLuPCF14Jnyib2m0Y6zF0DPJn8tRxB0SzOqjCEWadbWM8NJ1Ktk21yz3u0wwYYQ8im3OOGRbCjh55YvU%2FnPLssTLvs842vtSOaGoAEWDVidu556wvOjHHYlF6emwNcm5jRJ2DqZkOmi1yIb58qHLsFCYc7vFwOJbgfz3CF3WJxdeHc1RkL10fBCY27UwSM7f9wwvfPZmwY6ngF9IbIJNtcMBv4bpEtyvOkFnhi2%2FyhGU1de1wzhkAPqA33H1eOe7AIjQBG%2Fksr0xL48ES38YG7FAr2p5ybUjqdu5tGUE020lSoY4uCqAMqdi7UwJnQEBhEmhVWzDdPUnl2fYE%2Fk0D3XUxRvfFhgFAlIf1%2BtYp1Z3%2FuPzh5qaPyNYpGC%2BENYDKTW%2BB7MI7GbD%2FzyAtJpoFdTg8jLKgdk1w%3D%3D&Expires=1669140811"


wget -O "v1.0-trainval01_blobs.tgz" "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval01_blobs.tgz?AWSAccessKeyId=ASIA6RIK4RRMEER7HDIM&Signature=Qj2aXt1bSAQqFk7EAXZ11As5SB4%3D&x-amz-security-token=IQoJb3JpZ2luX2VjECsaCXVzLWVhc3QtMSJHMEUCIQCUSsn4qisaV18V1UO2tiDVSCA1VZyqSOkjdC03CNKQvAIgOPrs3hNMDwYp0lcVsdFHDAguLUVlYroQl2zNQh4XKVQq9AIIMxADGgw5OTkxMzk2MDk2ODgiDDy%2BVsEHwFLMc45ToyrRAqatoWIyg2Mi35HkDgIHNxUal16zPRD3tGyXwJSWqCWIsW%2BC9miqHEPqsJPVvlSjhM4j7J0SLMGwFjnJuOWeOyZwoN9D4FJ6lv0qdhOCpBp%2FkP9E0xwKME0RgftxK478Hmj%2FHMyJHW%2B4kqHd6%2FimJxZvNjF7ZSdA0KNyD7V%2BIrbdoQdufSTonEZqqUtVL0sQgAb3iGEUOm6xd2Re8P0krJC8YHhko7KKOf%2FdY%2FmgslwjR2%2FaSaqB3F%2B5pP5WX3DEvJLuPCF14Jnyib2m0Y6zF0DPJn8tRxB0SzOqjCEWadbWM8NJ1Ktk21yz3u0wwYYQ8im3OOGRbCjh55YvU%2FnPLssTLvs842vtSOaGoAEWDVidu556wvOjHHYlF6emwNcm5jRJ2DqZkOmi1yIb58qHLsFCYc7vFwOJbgfz3CF3WJxdeHc1RkL10fBCY27UwSM7f9wwvfPZmwY6ngF9IbIJNtcMBv4bpEtyvOkFnhi2%2FyhGU1de1wzhkAPqA33H1eOe7AIjQBG%2Fksr0xL48ES38YG7FAr2p5ybUjqdu5tGUE020lSoY4uCqAMqdi7UwJnQEBhEmhVWzDdPUnl2fYE%2Fk0D3XUxRvfFhgFAlIf1%2BtYp1Z3%2FuPzh5qaPyNYpGC%2BENYDKTW%2BB7MI7GbD%2FzyAtJpoFdTg8jLKgdk1w%3D%3D&Expires=1669140947"

wget -O "v1.0-trainval02_blobs.tgz" "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval02_blobs.tgz?AWSAccessKeyId=ASIA6RIK4RRMEER7HDIM&Signature=K%2Blx11JA1hS1l%2BvAdhqbKMfi2ho%3D&x-amz-security-token=IQoJb3JpZ2luX2VjECsaCXVzLWVhc3QtMSJHMEUCIQCUSsn4qisaV18V1UO2tiDVSCA1VZyqSOkjdC03CNKQvAIgOPrs3hNMDwYp0lcVsdFHDAguLUVlYroQl2zNQh4XKVQq9AIIMxADGgw5OTkxMzk2MDk2ODgiDDy%2BVsEHwFLMc45ToyrRAqatoWIyg2Mi35HkDgIHNxUal16zPRD3tGyXwJSWqCWIsW%2BC9miqHEPqsJPVvlSjhM4j7J0SLMGwFjnJuOWeOyZwoN9D4FJ6lv0qdhOCpBp%2FkP9E0xwKME0RgftxK478Hmj%2FHMyJHW%2B4kqHd6%2FimJxZvNjF7ZSdA0KNyD7V%2BIrbdoQdufSTonEZqqUtVL0sQgAb3iGEUOm6xd2Re8P0krJC8YHhko7KKOf%2FdY%2FmgslwjR2%2FaSaqB3F%2B5pP5WX3DEvJLuPCF14Jnyib2m0Y6zF0DPJn8tRxB0SzOqjCEWadbWM8NJ1Ktk21yz3u0wwYYQ8im3OOGRbCjh55YvU%2FnPLssTLvs842vtSOaGoAEWDVidu556wvOjHHYlF6emwNcm5jRJ2DqZkOmi1yIb58qHLsFCYc7vFwOJbgfz3CF3WJxdeHc1RkL10fBCY27UwSM7f9wwvfPZmwY6ngF9IbIJNtcMBv4bpEtyvOkFnhi2%2FyhGU1de1wzhkAPqA33H1eOe7AIjQBG%2Fksr0xL48ES38YG7FAr2p5ybUjqdu5tGUE020lSoY4uCqAMqdi7UwJnQEBhEmhVWzDdPUnl2fYE%2Fk0D3XUxRvfFhgFAlIf1%2BtYp1Z3%2FuPzh5qaPyNYpGC%2BENYDKTW%2BB7MI7GbD%2FzyAtJpoFdTg8jLKgdk1w%3D%3D&Expires=1669140988"