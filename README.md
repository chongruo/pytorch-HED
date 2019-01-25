# HED project
This work is an implementation of paper [Holistically-Nested Edge Detection](https://github.com/chongruo/my_configuration.git)


## Performance

On BSDS500

| Method | ODS (Fusion/Merged) | OIS (Fusion/Merged) | AP (Fusion/Merged) |
|:---|:---:|:---:|:---:| 
| My Implementation | 0.78731/0.78280 | 0.80623/0.80356 | 0.78632/0.83851 |
| Original Paper| 0.782/0.782 | 0.802/0.804 | 0.787/0.833 | 

As mentioned in the paper, Fusion refers to the fusion-output(dsn6) and Merged means results of combination of fusion layer and side outputs.

## How to Run
### Prerequisite:
* [AttrDict](https://github.com/bcj/AttrDict)

### Training 

