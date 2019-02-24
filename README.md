# HED in pytorch
This work is an implementation of paper [Holistically-Nested Edge Detection](https://github.com/chongruo/my_configuration.git).


## Performance

Input Image | dsn1 | dsn2  | dsn3  | dsn4 | dsn5  | Fusioned Output (dsn6)  | 
:-------------------------:|:----------------: | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: | 
![](https://github.com/chongruo/pytorch-HED/blob/master/images/3063/3063_input.png)  |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/3063/3063_dsn1.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/3063/3063_dsn2.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/3063/3063_dsn3.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/3063/3063_dsn4.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/3063/3063_dsn5.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/3063/3063_dsn6.png) | 
![](https://github.com/chongruo/pytorch-HED/blob/master/images/326025/326025_input.png)  |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/326025/326025_dsn1.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/326025/326025_dsn2.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/326025/326025_dsn3.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/326025/326025_dsn4.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/326025/326025_dsn5.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/326025/326025_dsn6.png) | 
![](https://github.com/chongruo/pytorch-HED/blob/master/images/100007/100007_input.png) | ![](https://github.com/chongruo/pytorch-HED/blob/master/images/100007/100007_dsn1.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/100007/100007_dsn2.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/100007/100007_dsn3.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/100007/100007_dsn4.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/100007/100007_dsn5.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/100007/100007_dsn6.png) | 
![](https://github.com/chongruo/pytorch-HED/blob/master/images/100039/100039_input.png)  |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/100039/100039_dsn1.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/100039/100039_dsn2.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/100039/100039_dsn3.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/100039/100039_dsn4.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/100039/100039_dsn5.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/100039/100039_dsn6.png) | 
![](https://github.com/chongruo/pytorch-HED/blob/master/images/103006/103006_input.png)  |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/103006/103006_dsn1.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/103006/103006_dsn2.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/103006/103006_dsn3.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/103006/103006_dsn4.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/103006/103006_dsn5.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/103006/103006_dsn6.png) | 
![](https://github.com/chongruo/pytorch-HED/blob/master/images/107072/107072_input.png)  |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/107072/107072_dsn1.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/107072/107072_dsn2.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/107072/107072_dsn3.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/107072/107072_dsn4.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/107072/107072_dsn5.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/107072/107072_dsn6.png) | 
![](https://github.com/chongruo/pytorch-HED/blob/master/images/109055/109055_input.png)  |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/109055/109055_dsn1.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/109055/109055_dsn2.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/109055/109055_dsn3.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/109055/109055_dsn4.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/109055/109055_dsn5.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/109055/109055_dsn6.png) | 
![](https://github.com/chongruo/pytorch-HED/blob/master/images/112056/112056_input.png)  |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/112056/112056_dsn1.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/112056/112056_dsn2.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/112056/112056_dsn3.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/112056/112056_dsn4.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/112056/112056_dsn5.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/112056/112056_dsn6.png) | 
![](https://github.com/chongruo/pytorch-HED/blob/master/images/130066/130066_input.png)  |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/130066/130066_dsn1.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/130066/130066_dsn2.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/130066/130066_dsn3.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/130066/130066_dsn4.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/130066/130066_dsn5.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/130066/130066_dsn6.png) | dsn refers to deep side output. 
![](https://github.com/chongruo/pytorch-HED/blob/master/images/15011/15011_input.png)  |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/15011/15011_dsn1.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/15011/15011_dsn2.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/15011/15011_dsn3.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/15011/15011_dsn4.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/15011/15011_dsn5.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/15011/15011_dsn6.png) | 
![](https://github.com/chongruo/pytorch-HED/blob/master/images/160067/160067_input.png)  |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/160067/160067_dsn1.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/160067/160067_dsn2.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/160067/160067_dsn3.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/160067/160067_dsn4.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/160067/160067_dsn5.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/160067/160067_dsn6.png) | 
![](https://github.com/chongruo/pytorch-HED/blob/master/images/16068/16068_input.png)  |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/16068/16068_dsn1.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/16068/16068_dsn2.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/16068/16068_dsn3.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/16068/16068_dsn4.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/16068/16068_dsn5.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/16068/16068_dsn6.png) | 
![](https://github.com/chongruo/pytorch-HED/blob/master/images/163004/163004_input.png)  |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/163004/163004_dsn1.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/163004/163004_dsn2.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/163004/163004_dsn3.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/163004/163004_dsn4.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/163004/163004_dsn5.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/163004/163004_dsn6.png) | 
![](https://github.com/chongruo/pytorch-HED/blob/master/images/220003/220003_input.png)  |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/220003/220003_dsn1.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/220003/220003_dsn2.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/220003/220003_dsn3.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/220003/220003_dsn4.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/220003/220003_dsn5.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/220003/220003_dsn6.png) | 
![](https://github.com/chongruo/pytorch-HED/blob/master/images/296058/296058_input.png)  |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/296058/296058_dsn1.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/296058/296058_dsn2.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/296058/296058_dsn3.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/296058/296058_dsn4.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/296058/296058_dsn5.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/296058/296058_dsn6.png) | 
![](https://github.com/chongruo/pytorch-HED/blob/master/images/41096/41096_input.png)  |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/41096/41096_dsn1.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/41096/41096_dsn2.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/41096/41096_dsn3.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/41096/41096_dsn4.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/41096/41096_dsn5.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/41096/41096_dsn6.png) | 
![](https://github.com/chongruo/pytorch-HED/blob/master/images/43051/43051_input.png)  |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/43051/43051_dsn1.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/43051/43051_dsn2.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/43051/43051_dsn3.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/43051/43051_dsn4.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/43051/43051_dsn5.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/43051/43051_dsn6.png) | 
![](https://github.com/chongruo/pytorch-HED/blob/master/images/48025/48025_input.png)  |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/48025/48025_dsn1.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/48025/48025_dsn2.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/48025/48025_dsn3.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/48025/48025_dsn4.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/48025/48025_dsn5.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/48025/48025_dsn6.png) |<br>
![](https://github.com/chongruo/pytorch-HED/blob/master/images/2018/2018_input.png)  |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/2018/2018_dsn1.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/2018/2018_dsn2.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/2018/2018_dsn3.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/2018/2018_dsn4.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/2018/2018_dsn5.png) |  ![](https://github.com/chongruo/pytorch-HED/blob/master/images/2018/2018_dsn6.png) | 

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

