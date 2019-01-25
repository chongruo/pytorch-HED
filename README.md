# HED in pytorch
This work is an implementation of paper [Holistically-Nested Edge Detection](https://github.com/chongruo/my_configuration.git).


## Performance

Input Image | dsn1 | dsn2  | dsn3  | dsn4 | dsn5  | Fusioned Output (dsn6)  | 
:-------------------------:|:----------------: | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: | 
![](https://github.com/chongruo/pytorch-HED/blob/master/images/3063/3063_input.png)  |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/3063/3063_dsn1.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/3063/3063_dsn2.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/3063/3063_dsn3.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/3063/3063_dsn4.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/3063/3063_dsn5.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/3063/3063_dsn6.png) | 
![](https://github.com/chongruo/pytorch-HED/blob/master/images/2018/2018_input.png)  |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/2018/2018_dsn1.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/2018/2018_dsn2.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/2018/2018_dsn3.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/2018/2018_dsn4.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/2018/2018_dsn5.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/2018/2018_dsn6.png) | 

dsn refers to deep side output. 

<br>

**On BSDS500**

| Method | ODS (Fusion/Merged) | OIS (Fusion/Merged) | AP (Fusion/Merged) |
|:---|:---:|:---:|:---:| 
| My Implementation | 0.78731/0.78280 | 0.80623/0.80356 | 0.78632/0.83851 |
| Original Paper| 0.782/0.782 | 0.802/0.804 | 0.787/0.833 | 

As mentioned in the paper, Fusion refers to the fusion-output(dsn6) and Merged means results of combination of fusion layer and side outputs.

<br>

## How to Run
### Prerequisite:
* [AttrDict](https://github.com/bcj/AttrDict)

### Training 

