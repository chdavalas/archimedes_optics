# archimedes_optics
Current status: 

* Item ARNIQA is right most of the time in terms of image quality, without training (kadid10k backbone) by using the average we can detect possible drift although is not fully tested yet. In some videos arniqa suggests issues without distortions.

* Item LSTM is occasionally right without training (pretty unstable) without training (using kadid10k backbone with kadid10k lstm pretrained).

* Item DDRIFT cannot show results in video despite working very well in kadid10k without training (using arniqa kadid10k backbone in all examples). Trying to calibrate with video with no progress so far. Possible training with custom dataset (drone images, labels) where we can use a disjoint dataset by camera footage, create labels according to clean footage with artificial distortions. It is a drawback but can be justified.

The results below are a decent effort so far for the factory inspection video

<div align="center">
	<img src="current_results.jpg">
</div>