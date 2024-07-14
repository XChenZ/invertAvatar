## (Under Contruction) InvertAvatar: Incremental GAN Inversion for Generalized Head Avatars

![Teaser image](./assets/teaser.png)

**InvertAvatar: Incremental GAN Inversion for Generalized Head Avatars**<br>
[Xiaochen Zhao*](https://xiaochen-zhao.netlify.app/), [Jingxiang Sun*](https://mrtornado24.github.io/), [Lizhen Wang](https://lizhenwangt.github.io/), Jinli Suo, [Yebin Liu](http://www.liuyebin.com/)<br><br>


[**Project**](https://xchenz.github.io/invertavatar_page/) | [**Paper**](https://arxiv.org/abs/2312.02222)

Abstract: *While high fidelity and efficiency are central to the creation of digital head avatars, recent methods relying on 2D or 3D generative models often experience limitations such as shape distortion, expression inaccuracy, and identity flickering. Additionally, existing one-shot inversion techniques fail to fully leverage multiple input images for detailed feature extraction. We propose a novel framework, \textbf{Incremental 3D GAN Inversion}, that enhances avatar reconstruction performance using an algorithm designed to increase the fidelity from multiple frames, resulting in improved reconstruction quality proportional to frame count. Our method introduces a unique animatable 3D GAN prior with two crucial modifications for enhanced expression controllability alongside an innovative neural texture encoder that categorizes texture feature spaces based on UV parameterization. Differentiating from traditional techniques, our architecture emphasizes pixel-aligned image-to-image translation, mitigating the need to learn correspondences between observation and canonical spaces. Furthermore, we incorporate ConvGRU-based recurrent networks for temporal data aggregation from multiple frames, boosting geometry and texture detail reconstruction. The proposed paradigm demonstrates state-of-the-art performance on one-shot and few-shot avatar animation tasks.*
