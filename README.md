# LaNet
LaNet - Unidirectional LSTM architecture for lane identification using Accelerometer data

Reference Codebase for LaNet proposal submitted to ACM BuildSys 2019.

Data for model training available at : https://drive.google.com/file/d/1keAI9dy1iqarq84IJfWamleNDFrEMXkd/view?usp=sharing

![LaNet - Deep LSTM Model](https://github.com/SaiVineethKalluruSrinivas/LaNet/blob/master/LSTMArch.jpg)

## Highlights
- Codes for LaNet LSTM arch along with WeightedLSTMCrossEntropyLoss are included.
- Supports Tensorboard visualisations to visualize incremental accuracy observed in each LSTM Cell
- Provides scope for reconfiguration of model parameters including sub-drive length and sub-segment lengths.

## Default model configurations

- Sample Length "l"                  = 800000
- Sub-drive stride "s"               = 50000
- Sub-segment length"d"              = 50000
- Sub-segment stride "m"             = int(50000/2)
- LSTM Hidden dimension "H"          = 300
- Learning rate "lr"                 = 0.005
- Num epochs                         = 4
- Number of LSTM Layers              = 2
- Number of lanes/num_classes        = 2
- Accelerometer data "sampling_rate" = 2000
- Batch Size = 512



