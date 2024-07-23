# archimedes_optics
Current status (Video testing): 

* ARNIQA is right most of the time in terms of image quality, without training (kadid10k backbone) by using the average we can detect possible drift although is not fully tested yet. In some videos arniqa suggests issues without distortions. It can always provide image quality (unlike LSTM and DDRIFT) without training, although drift detection requires more justification. This method is very good in factory, traffic, and pipe inspection.

* LSTM is occasionally right without training (although recall is being mediocre) without training (using kadid10k backbone with kadid10k lstm pretrained). We possibly need custom LSTM training in a dataset. Tests with custom training are a bit better. This method is also good in traffic and pipe inspection.

* DDRIFT works with the ARNIQA (and any other) backbone when we use a random crop transformation in order to provide variety in the reference and target distribution. This dataloader transformation does not affect the other methods in an serious manner, this method is slightly better in assembly line inspection.

In order to ensure the optimal performance, an integration of randomCrop function with the DDRIFT method is being considered, in order to make the other methods "dataloader indepedent", cropping and resizing transformations might have to be in integrated per model. 

<div align="center">
	<img src="current_results.jpg">
</div>