Code for final ML project CS2750. Please follow the below instructions to run the code.

1. All metadata can be download from https://drive.google.com/drive/folders/1DuH0YaEox08ZwzZDpRMOaFpMCeRyxiEF
    Please download only multimodal samples.

2. All image data can be downloaded from https://drive.google.com/file/d/1cjY6HsHaSZuLVHywIxD5xQqng33J5S2b/view

3. All comment data can be downloaded from https://drive.google.com/drive/folders/150sL4SNi5zFK8nmllv5prWbn0LyvLzvo

4. Run the file resnet.py : this code loads the image dataset for the image only unimodal and starts the finetuning of the Pretrained Resnet50 model. The model is trained for 20 epochs.
The model is then evaluated against a test dataset and the results are recorded.

5. Run the file bert.py : This code loads the text dataset for the textual unimodal and stats the finetuning of the pretrained Bert model. This model is trained for 2 epochs.
The model is then evaluated against a test dataset and the results are recorded.

6. Run the file late_fusion.py : This code loads the hybrid (Image + text) dataset and stats the late fusion model training. The models are step 4 and 5 are used as feature extractors after freezing their parameters and then concatenated before using a linear layer. 