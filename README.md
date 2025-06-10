# Offical repo for "Cheaper is Better: A Discount-Aware Network for Conversion Rate Prediction in E-commerce Recommendation System"
The implementation of our proposed method is based on our company-customized distributed Tensorflow framework for better industrial applications. Releasing code related to the framework is not allowed and detailed information about input features are also confidential. So with these information removed, part of source code is extracted and simplified as supplementary materials. The released source code is not runnable but able to illustrate the implementation of model structure.

## DANet Recommend
This is the tensorflow implementation of DANet. This model introduces the concept of discount rate in CVR modeling for the first time. The main innovations are as follows:
1. we introduce a modeling method of personalized 𝐷𝑖𝑠𝑐𝑜𝑢𝑛𝑡 𝑅𝑎𝑡𝑒(𝐷𝑅) in CVR prediction which achieve high prediction accuracy by self-adapting to different periods (e.g. flat sale period and promotion period).
2. We propose a novel Discount-Aware Network(DANet), which includes a time-frequency transformation module, a distribution correction module, and a supervised regression auxiliary task, allowing for a more comprehensive understanding of long-term DR trends of item and users’ sensitivity to DR.

## Project File Description
`config/algo.conf` is the model parameter configuration, including multi-behavior sequence sideinfo, parameters of each network layer, activation function, optimizer parameter settings, loss fusion parameters, etc.

`model/model.py` is the network structure of DANet. The graph construction function is `build_graph()`, which includes the `inference()` describing the network forward process, the multi-party `loss()`, and the `optimizer()`.

![image](https://github.com/user-attachments/assets/f622c5e9-1ba3-4cbb-b32d-c0824a10cf43)



