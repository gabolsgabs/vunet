# Content Based Singing Voice Source Separation via Strong Conditioning Using Aligned Phonemes.

## ABSTRACT

Informed source separation has recently gained renewed interest with the introduction of neural networks and the availability of large multitrack datasets containing both the mixture and the separated sources.These approaches use prior information about the target source to improve separation.Historically, Music Information Retrieval ({MIR}) researchers have focused primarily on score-informed source separation, but more recent approaches explore lyrics-informed source separation.However, because of the lack of multitrack datasets with time-aligned lyrics, models use weak conditioning with the non-aligned lyrics.In this paper, we present a multimodal multitrack dataset with lyrics aligned in time at the word level with phonetic information as well as explore strong conditioning using the aligned phonemes.Our model follows a {U-Net} architecture and takes as input both the magnitude spectrogram of a musical mixture and a matrix with aligned phoneme information.The phoneme matrix is embedded to obtain the parameters that control Feature-wise Linear Modulation ({FiLM}) layers.These layers condition the {U-Net} feature maps to adapt the separation process to the presence of different phonemes via affine transformations.We show that phoneme conditioning can be successfully applied to improve singing voice source separation.

You can find a detailed explanation at [https://program.ismir2020.net/poster_6-07.html](https://program.ismir2020.net/poster_6-07.html)

Cite this paper:

>@inproceedings{Meseguer-Brocal_2020,
	Author = {Meseguer-Brocal, Gabriel and Peeters, Geoffroy},
	Booktitle = {21th International Society for Music Information Retrieval Conference},
	Editor = {ISMIR},
	Month = {October},
	Title = {Content Based Singing Voice Source Separation via Strong Conditioning Using Aligned Phonemes.},
	Year = {2020}}

#### DATASET

SOON


### CODE

SOON: Not stable version yet!

#### How to use this package:

SOON


### CONTACT

You can contact us at:

  > gabriel dot meseguerbrocal at ircam dot fr
